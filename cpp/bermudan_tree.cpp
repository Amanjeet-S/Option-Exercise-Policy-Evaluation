// Author: Amanjeet Singh.
// Native Cox-Ross-Rubinstein rollback on a uniform positive exercise calendar.
// The model is established; finite-grid values are not diffusion certificates.
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace pricing {
constexpr std::size_t maximum_steps = 8192;

void finite(double value, const char* label) {
    if (!std::isfinite(value)) throw std::invalid_argument(std::string(label) + " is nonfinite");
}

double represented(double value, const char* label) {
    finite(value, label);
    if (std::fpclassify(value) == FP_SUBNORMAL)
        throw std::underflow_error(std::string(label) + " is subnormal; arithmetic range unsupported");
    return value;
}

double multiply(double left, double right, const char* label) {
    const double value = represented(left * right, label);
    if (left != 0.0 && right != 0.0 && value == 0.0)
        throw std::underflow_error(std::string(label) + " underflowed to zero");
    return value;
}

double divide(double numerator, double denominator, const char* label) {
    if (denominator == 0.0) throw std::invalid_argument(std::string(label) + " divides by zero");
    const double value = represented(numerator / denominator, label);
    if (numerator != 0.0 && value == 0.0)
        throw std::underflow_error(std::string(label) + " underflowed to zero");
    return value;
}

double difference(double left, double right, const char* label) {
    return represented(left - right, label);
}

double positive_exp(double exponent, const char* label) {
    finite(exponent, label);
    const double value = represented(std::exp(exponent), label);
    if (!(value > 0.0)) throw std::underflow_error(std::string(label) + " underflowed to zero");
    return value;
}

struct Market {
    const double spot, rate, dividend, volatility;
    Market(double s, double r, double q, double sigma)
        : spot(s), rate(r), dividend(q), volatility(sigma) {
        represented(spot, "Spot"); finite(rate, "Rate"); finite(dividend, "Dividend yield");
        finite(volatility, "Volatility");
        if (!(spot > 0.0) || volatility < 0.0)
            throw std::invalid_argument("Spot must be positive and volatility nonnegative");
    }
};

enum class Kind { call, put };

struct Option {
    const double strike, maturity;
    const Kind kind;
    Option(double k, double t, Kind type) : strike(k), maturity(t), kind(type) {
        represented(strike, "Strike"); represented(maturity, "Maturity");
        if (!(strike > 0.0) || !(maturity > 0.0))
            throw std::invalid_argument("Strike and maturity must be strictly positive");
    }
    double payoff(double stock) const {
        const double intrinsic = kind == Kind::call ? difference(stock, strike, "Intrinsic reward")
                                                    : difference(strike, stock, "Intrinsic reward");
        return std::max(intrinsic, 0.0);
    }
};

struct UniformCalendar {
    const std::size_t steps, rights, stride;
    UniformCalendar(std::size_t n, std::size_t m) : steps(n), rights(m), stride(m ? n / m : 0) {
        if (steps == 0 || rights == 0) throw std::invalid_argument("Steps and rights must be positive integers");
        if (steps > maximum_steps || rights > maximum_steps)
            throw std::length_error("Steps and rights exceed the 8192 resource cap");
        if (rights > steps || steps % rights != 0)
            throw std::invalid_argument("Uniform exercise rights must align exactly: steps % rights == 0");
    }
    bool permits(std::size_t index) const { return index > 0 && index % stride == 0; }
};

struct Result {
    double value;
    bool deterministic;
    double probability;
};

double stock_node(const Market& market, std::size_t index, std::size_t up_count, double log_up) {
    const double exponent = multiply(2.0 * static_cast<double>(up_count) - static_cast<double>(index),
                                     log_up, "Stock-node exponent");
    return multiply(market.spot, positive_exp(exponent, "Stock-node multiplier"), "Stock node");
}

Result price(const Market& market, const Option& option, const UniformCalendar& calendar) {
    const double duration = divide(option.maturity, static_cast<double>(calendar.steps), "Time step");
    if (!(duration > 0.0)) throw std::underflow_error("Time step underflowed to zero");
    const double carry = difference(market.rate, market.dividend, "Carry rate");
    if (market.volatility == 0.0) {
        double best = 0.0;
        for (std::size_t right = 1; right <= calendar.rights; ++right) {
            const double time = multiply(option.maturity,
                static_cast<double>(right) / static_cast<double>(calendar.rights), "Exercise time");
            const double stock = multiply(market.spot,
                positive_exp(multiply(carry, time, "Deterministic stock exponent"), "Deterministic multiplier"),
                "Deterministic stock");
            const double discount = positive_exp(multiply(-market.rate, time, "Discount exponent"), "Discount");
            best = std::max(best, multiply(discount, option.payoff(stock), "Discounted reward"));
        }
        return {best, true, 0.0};
    }
    const double log_up = multiply(market.volatility, std::sqrt(duration), "Log up multiplier");
    if (!(log_up > 0.0)) throw std::underflow_error("Nonzero volatility produced a zero multiplier");
    const double up = positive_exp(log_up, "Up multiplier");
    const double down = positive_exp(-log_up, "Down multiplier");
    const double denominator = difference(up, down, "CRR denominator");
    if (!(denominator > 0.0)) throw std::invalid_argument("Volatility is too small to resolve the CRR denominator");
    const double forward = positive_exp(multiply(carry, duration, "Forward exponent"), "Forward multiplier");
    const double probability = divide(difference(forward, down, "CRR numerator"), denominator, "CRR probability");
    if (probability < 0.0 || probability > 1.0)
        throw std::invalid_argument("CRR no-arbitrage probability is outside [0,1]; increase steps");
    const double discount = positive_exp(multiply(-market.rate, duration, "Discount exponent"), "Discount");
    std::vector<double> values(calendar.steps + 1);
    for (std::size_t i = 0; i <= calendar.steps; ++i)
        values[i] = option.payoff(stock_node(market, calendar.steps, i, log_up));
    for (std::size_t level = calendar.steps; level > 0; --level) {
        const std::size_t index = level - 1;
        for (std::size_t i = 0; i <= index; ++i) {
            const double low = multiply(1.0 - probability, values[i], "Down expectation contribution");
            const double high = multiply(probability, values[i + 1], "Up expectation contribution");
            const double expected = represented(low + high, "Continuation expectation");
            double value = multiply(discount, expected, "Discounted continuation");
            if (calendar.permits(index))
                value = std::max(value, option.payoff(stock_node(market, index, i, log_up)));
            values[i] = value;
        }
        values.resize(level);
    }
    return {values.front(), false, probability};
}
} // namespace pricing

namespace interface {
double real(const char* text, const char* label) {
    const std::string value(text);
    if (value.empty() || value.find_first_of(" \t\n\r\f\v") != std::string::npos)
        throw std::invalid_argument(std::string(label) + " requires one numeric token");
    char* end = nullptr; errno = 0;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE)
        throw std::invalid_argument(std::string(label) + " is invalid or outside binary64 range");
    pricing::finite(parsed, label);
    return parsed;
}

std::size_t integer(const char* text, const char* label) {
    const std::string value(text);
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument(std::string(label) + " requires an unsigned integer token");
    return parsed;
}

std::string escaped(const std::string& text) {
    std::string result;
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') { result += '\\'; result += static_cast<char>(c); }
        else if (c >= 32) result += static_cast<char>(c);
        else result += '?';
    }
    return result;
}
} // namespace interface

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    try {
        if (argc != 10)
            throw std::invalid_argument("Usage: bermudan_tree spot strike maturity rate dividend volatility call|put steps rights");
        const pricing::Market market(interface::real(argv[1], "Spot"), interface::real(argv[4], "Rate"),
            interface::real(argv[5], "Dividend yield"), interface::real(argv[6], "Volatility"));
        const std::string kind(argv[7]);
        if (kind != "call" && kind != "put") throw std::invalid_argument("Option kind must be call or put");
        const pricing::Option option(interface::real(argv[2], "Strike"), interface::real(argv[3], "Maturity"),
            kind == "call" ? pricing::Kind::call : pricing::Kind::put);
        const pricing::UniformCalendar calendar(interface::integer(argv[8], "Steps"), interface::integer(argv[9], "Rights"));
        const auto result = pricing::price(market, option, calendar);
        std::cout << std::setprecision(17) << "{\"ok\":true,\"author\":\"Amanjeet Singh\",\"value\":" << result.value
                  << ",\"steps\":" << calendar.steps << ",\"exercise_rights\":" << calendar.rights
                  << ",\"time_zero_exercise\":false,\"zero_volatility\":" << (result.deterministic ? "true" : "false")
                  << ",\"probability\":";
        if (result.deterministic) std::cout << "null"; else std::cout << result.probability;
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"ok\":false,\"author\":\"Amanjeet Singh\",\"error\":\"" << interface::escaped(error.what()) << "\"}\n";
        return 2;
    }
}
