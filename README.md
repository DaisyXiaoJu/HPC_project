# HPC_project

1.所有commit为“Files from the old repository”的文件和文件夹均为原二维问题的文件，仅做保留用 \\
2.显式的文件保存在heat_tran_explicit文件夹中，隐式文件保存在heat_tran_implicit中，copy后输入命令：
$ make clean
$ make
$ bsub < ty_script.sh
(如果$ make出错，可能需要修改Makefile第3行 PETSC_DIR := /work/mae-zhonghc/lib/petsc-3.16.6-opt 的路径)
3.生成的.h5文件打开后有两个dataset，uout存储当前迭代步温度，para存储时间，网格分辨率和时间步长信息
4.检视.log文件查看更多输出信息
5.太乙工作路径：/work/mae-zhonghc/hpc/HPC_project，同样保存了上述两个文件夹和原先的旧文件
6.原repository链接：https://github.com/GBoy2509/HPC_project ，保留所有commits
