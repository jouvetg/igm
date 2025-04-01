   50  nsys profile --gpu-metrics-device=all --trace nvtx,cuda,cudnn igm_run +experiment=params
   51  sudo nsys profile --gpu-metrics-device=all --trace nvtx,cuda,cudnn igm_run +experiment=params
   52  nsys status -e
   53  cat /etc/modprobe.d
   54  cd /etc/modprobe.d
   55  ls
   56  nano nvidia_nsight_permission.conf
   57  sudo nano nvidia_nsight_permission.conf
   58  cd /home/bfinley/Documents/IGM/igm/tests/examples/synthetic
   59  nsys status -e

	https://forums.developer.nvidia.com/t/cant-get-gpu-metrics-with-nsight-system/305328/9
   https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
   
   WORKED (but nsys status -e still shows root disabled...):
   https://forums.developer.nvidia.com/t/unable-to-get-gpu-profile-on-linux-with-cap-sys-admin-enabled-user/174232

      45  update-initramfs -u -k all
   46  sudo update-initramfs -u -k all
   47  nsys status -e
   48  modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
   49  nsys status -e
   50  which nsys
   51  nsys --version
   52  modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
   53  nsys status -e
   54  cat /etc/modprobe.d/nvidia_nsight_permission.conf
   55  ls /etc/modprobe.d
   56  sudo nano /etc/modprobe.d/nvidia-prof.conf
   57  ls /etc/modprobe.d/
   59  sudo update-initramfs -u -k all
   60  sudo update-grub

   SUDO:
   https://askubuntu.com/questions/611528/why-cant-sudo-find-a-command-after-i-added-it-to-path

   # PINNED MEMORY IN TF
   (not sure if it worked though as I still get it)
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto


   `nsys profile --gpu-metrics-device=all --trace nvtx,cuda,cudnn --cudabacktrace=all --cuda-graph-trace=graph --cuda-memory-usage=true igm_run +experiment=params`

   nsys profile --gpu-metrics-device=all --trace nvtx,cuda,cudnn --cudabacktrace=all --cuda-graph-trace=graph igm_run +experiment=params