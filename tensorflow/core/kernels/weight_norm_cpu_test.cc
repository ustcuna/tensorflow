#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

enum class Device { CPU };

class WeightNormTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType data_type) {
    // weight_norm op.
    TF_CHECK_OK(NodeDefBuilder("weight_norm", "WeightNorm")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT64))
                  .Attr("T", data_type)
                  .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

//----------------------------------------------------------------------------//
//                            Accuracy Check                                  //
//----------------------------------------------------------------------------//
TEST_F(WeightNormTest, Norm_Test_All_Dim) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 2; });

  // L2 norm should be 35.21363372331802
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.02839809;
    output_array[1] = 0.05679618;
    output_array[2] = 0.08519428;
    output_array[3] = 0.11359237;
    output_array[4] = 0.14199046;
    output_array[5] = 0.17038855;
    output_array[6] = 0.19878664;
    output_array[7] = 0.22718473;
    output_array[8] = 0.25558283;
    output_array[9] = 0.28398092;
    output_array[10] = 0.31237901;
    output_array[11] = 0.3407771;
    output_array[12] = 0.36917519;
    output_array[13] = 0.39757328;
    output_array[14] = 0.42597138;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

TEST_F(WeightNormTest, Norm_Test_Dim0) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 0; });

  // L2 norm should be
  // [12.56980509 14.03566885 15.55634919 17.11724277 18.70828693]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.07955573;
    output_array[1] = 0.1424941;
    output_array[2] = 0.1928473;
    output_array[3] = 0.2336825;
    output_array[4] = 0.26726124;
    output_array[5] = 0.47733437;
    output_array[6] = 0.49872935;
    output_array[7] = 0.51425948;
    output_array[8] = 0.52578561;
    output_array[9] = 0.53452248;
    output_array[10] = 0.87511301;
    output_array[11] = 0.8549646;
    output_array[12] = 0.83567165;
    output_array[13] = 0.81788873;
    output_array[14] = 0.80178373;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

TEST_F(WeightNormTest, Norm_Test_Dim1) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 1; });

  // L2 norm should be
  // [ 7.41619849 18.16590212 29.24038303]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.13483997;
    output_array[1] = 0.26967993;
    output_array[2] = 0.4045199;
    output_array[3] = 0.53935986;
    output_array[4] = 0.67419983;
    output_array[5] = 0.33028913;
    output_array[6] = 0.38533732;
    output_array[7] = 0.44038551;
    output_array[8] = 0.49543369;
    output_array[9] = 0.55048188;
    output_array[10] = 0.37619206;
    output_array[11] = 0.41039134;
    output_array[12] = 0.44459062;
    output_array[13] = 0.4787899;
    output_array[14] = 0.51298918;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

//----------------------------------------------------------------------------//
//                       Performance benchmarks                               //
//----------------------------------------------------------------------------//
template <typename T>
static Graph* WeightNormGraph(const int64& dim, const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());

  DataType data_type = DataTypeToEnum<T>::v();

  Tensor input_t(data_type, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  
  Tensor norm_dim(DT_INT64, TensorShape({}));
  norm_dim.scalar<int64>()() = dim;
  
  Node* weightnorm;
  // weight_norm op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("weight_norm"), "WeightNorm")
                  .Input(input)
                  .Input(test::graph::Constant(graph, norm_dim))
                  .Attr("T", data_type)
                  .Finalize(graph, &weightnorm));

  return graph;
}

#define BM_WEIGHTNORM(A, B, type, dim, T)                               \
  static void BM_WEIGHTNORM_##A##_##B##_##type##_##dim##_##T(           \
      ::testing::benchmark::State& state) {                             \
    int64 num_computed_elements = (A) * (B);                            \
    int64 flops_per_iter = num_computed_elements;                       \
    test::Benchmark(#type, WeightNormGraph<T>(dim, {A, B}))             \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_WEIGHTNORM_##A##_##B##_##type##_##dim##_##T)->UseRealTime();

#define BENCHMARK_WEIGHTNORM(A, B, type, T)    \
  BM_WEIGHTNORM(A, B, type, 2, T);             \
  BM_WEIGHTNORM(A, B, type, 0, T);             \
  BM_WEIGHTNORM(A, B, type, 1, T);

#define BENCHMARK_DTYPE(T)                     \
  BENCHMARK_WEIGHTNORM(32, 64, cpu, T);        \
  BENCHMARK_WEIGHTNORM(64, 128, cpu, T);       \
  BENCHMARK_WEIGHTNORM(128, 256, cpu, T);      \
  BENCHMARK_WEIGHTNORM(512, 256, cpu, T);      \
  BENCHMARK_WEIGHTNORM(1024, 512, cpu, T);     \
  BENCHMARK_WEIGHTNORM(1024, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(2048, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(3072, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(4096, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(8192, 4096, cpu, T);    \
  BENCHMARK_WEIGHTNORM(16384, 8192, cpu, T);   \
  BENCHMARK_WEIGHTNORM(24576, 16384, cpu, T);

BENCHMARK_DTYPE(float)
BENCHMARK_DTYPE(double)

}  // namespace
} // namespace tensorflow