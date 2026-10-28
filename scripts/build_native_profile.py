"""Build a π0 native library with Hazy VM instruction timing enabled."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT=Path(__file__).resolve().parents[1]
SOURCES=('model.cu','native.cu','dense.cu','tcgen_probe.cu','attention.cu',
         'norm.cu','model_attention.cuh','launch_trace.cpp')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pi0-native-profile-') as directory:
        folder=Path(directory)
        for name in SOURCES:shutil.copy2(ROOT/'native_pi0'/name,folder/name)
        config=folder/'tcgen_probe.cu';source=config.read_text()
        marker='struct ComputeConfig:megakernel::default_config {'
        if source.count(marker)!=1:raise RuntimeError('ComputeConfig declaration changed')
        config.write_text(source.replace(marker,marker+'\n    static constexpr bool TIMING_RECORD_ENABLED=true;',1))
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a',
                        '-DKITTENS_HOPPER','-DKITTENS_BLACKWELL','-DNATIVE_COMPUTE_CONSUMER_WARPS=4',
                        '--expt-relaxed-constexpr','--expt-extended-lambda',
                        '-I'+str(ROOT/'third_party/Megakernels/include'),
                        '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
                        '-shared','-Xcompiler','-fPIC',str(folder/'model.cu'),str(folder/'launch_trace.cpp'),
                        '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64',
                        '-lcuda','-o',str(output)],check=True)


if __name__=='__main__':main()
