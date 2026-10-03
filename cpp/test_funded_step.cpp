// Standalone mathematical checks, also used with address/undefined sanitizers.
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

extern "C" int qf_funded_advance(const double*, std::size_t, const double*,
    const double*, const double*, const double*, const double*, std::size_t, double*) noexcept;

int main() {
    for (std::size_t n : {1U, 2U, 31U, 600U}) {
        std::vector<double> d(n), initial(n+2), output(n+2);
        for (std::size_t i = 0; i < n; ++i) d[i] = .3*(i+1)*(i+1);
        for (std::size_t i = 0; i < n+2; ++i) initial[i] = 1.0+2.0*i/(n+1);
        const double widths[]{.01,.03,.02}, weights[]{1.0,.5,.5};
        const double left[]{1.,1.,1.}, right[]{3.,3.,3.};
        assert(qf_funded_advance(d.data(),n,initial.data(),widths,weights,left,right,3,output.data())==0);
        for (std::size_t i = 0; i < n+2; ++i) assert(std::abs(output[i]-initial[i])<1e-11);
    }
    double d[]{1.}, v[]{0.,1.,0.}, h[]{.1}, w[]{1.}, b[]{0.}, out[3];
    assert(qf_funded_advance(nullptr,1,v,h,w,b,b,1,out)==1);
    assert(qf_funded_advance(d,0,v,h,w,b,b,1,out)==1);
    assert(qf_funded_advance(d,1,v,h,w,b,b,0,out)==1);
    d[0]=-1.; assert(qf_funded_advance(d,1,v,h,w,b,b,1,out)==1);
    d[0]=std::numeric_limits<double>::max(); h[0]=d[0];
    assert(qf_funded_advance(d,1,v,h,w,b,b,1,out)==2);
}
