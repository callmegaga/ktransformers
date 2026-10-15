// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -fp-model=precise -O2 bench_sycl_limits.cpp -o bench_sycl_limits
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:gpu KT_SYCL_PROFILE=1 ./bench_sycl_limits
// Diagnostic roofline probes, NOT end-to-end inference predictions.
#include "../operators/sycl/gptq_int4_xmx-kernels.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace sycl_xmx;
double elapsed(sycl::event e) { e.wait_and_throw();return (e.get_profiling_info<sycl::info::event_profiling::command_end>()-e.get_profiling_info<sycl::info::event_profiling::command_start>())/1.e9; }
double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
template<int Width> void memory_probe() {
  constexpr size_t bytes=256ul*1024*1024, count=bytes/sizeof(uint32_t);
  USM<uint32_t> in,out;in.reserve(count);out.reserve(count);
  auto& q=sycl_int4::queue();q.memset(in.data,0x67,bytes).wait_and_throw();q.memset(out.data,0,bytes).wait_and_throw();
  auto *a=in.data,*b=out.data;std::vector<double> times;
  for(int r=-3;r<20;++r) {
    double t=elapsed(q.parallel_for(sycl::range<1>(count/Width), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      es::simd<uint32_t,Width> v;v.copy_from(a+id[0]*Width);v.copy_to(b+id[0]*Width);
    }));if(r>=0)times.push_back(t);
  }
  q.memcpy(in.data,out.data,sizeof(uint32_t)).wait_and_throw();
  if(in.data[0]!=0x67676767u)throw std::runtime_error("streaming copy mismatch");
  std::cout<<"stream width_dwords="<<Width<<" seconds="<<median(times)<<" aggregate_GBps="<<(2.*bytes/median(times)/1.e9)<<std::endl;
}
void compute_probe() {
  constexpr int work=4096, iterations=256;
  USM<int8_t>a;USM<uint32_t>b;USM<int>out;a.reserve(work*256);b.reserve(work*64);out.reserve(work*128*4);
  for(int j=0;j<work*256;++j)a.data[j]=(j%17)-8;
  for(int j=0;j<work*64;++j)b.data[j]=0x01234567u+j;
  auto *ap=a.data;auto *bp=b.data;auto*op=out.data;std::vector<double> times;
  for(int r=-3;r<20;++r) {
    double t=elapsed(sycl_int4::queue().parallel_for(sycl::range<1>(work), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      es::simd<int8_t,256> av;av.copy_from(ap+id[0]*256);
      es::simd<uint32_t,64> bv;bv.copy_from(bp+id[0]*64);
      es::simd<int,128> c0=0,c1=1,c2=2,c3=3;
      for(int k=0;k<iterations;++k) {
#define DPAS(C) C=xmx::dpas<8,8,int,int,uint32_t,int8_t,xmx::dpas_argument_type::s4,xmx::dpas_argument_type::s8>(C,bv,av)
        DPAS(c0);DPAS(c1);DPAS(c2);DPAS(c3);
#undef DPAS
      }
      c0.copy_to(op+id[0]*512);c1.copy_to(op+id[0]*512+128);c2.copy_to(op+id[0]*512+256);c3.copy_to(op+id[0]*512+384);
    }));if(r>=0)times.push_back(t);
  }
  for (int row = 0; row < 8; ++row) for (int col = 0; col < 16; ++col) {
    int expected = 0;
    for (int k = 0; k < 32; ++k) {
      int w = (b.data[(k / 8) * 16 + col] >> ((k % 8) * 4)) & 15;
      if (w >= 8) w -= 16;
      expected += a.data[row * 32 + k] * w;
    }
    if (out.data[row * 16 + col] != expected * iterations)
      throw std::runtime_error("DPAS scalar-reference mismatch");
  }
  for(int j=0;j<128;++j)if(out.data[j]+1!=out.data[128+j]||out.data[j]+2!=out.data[256+j]||out.data[j]+3!=out.data[384+j])throw std::runtime_error("DPAS result mismatch");
  const double operations=double(work)*iterations*4*8*16*32*2;
  std::cout<<"S8xS4_dpas seconds="<<median(times)<<" effective_TOPS="<<operations/median(times)/1.e12<<std::endl;
}
int main(){try{std::cout<<sycl_int4::queue().get_device().get_info<sycl::info::device::name>()<<std::endl;compute_probe();memory_probe<64>();memory_probe<256>();}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
