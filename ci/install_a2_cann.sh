#!/usr/bin/bash

rm -rf cann
mkdir -p cann && cd cann
curl -O https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-aarch64.run
curl -O https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run
curl -O https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-aarch64.run


if [[ $? -ne 0 ]]; then
  echo "Failed to download"
  exit 1
fi

chmod +x Ascend-cann*.run
./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend && source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo "toolkit install success"
./Ascend-cann-910b*.run --install --quiet --install-path=/usr/local/Ascend && echo "ops install success"
./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend && source /usr/local/Ascend/nnal/atb/set_env.sh && echo "nnal install success"
rm -rf *
