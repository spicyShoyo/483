# 483 Final Project
This is the repository of our final project for Applied Parallel Programming course at U of I.

This program performs PCA-KNN to decode a check.
The program runs on machines with a Nvidia GPU and cuda-toolkit installed.

Compile: 
	 
	 make all // For PCA-KNN
	 
	 make time // For timing PCA-KNN	

	 make debug // For debug
	
	 make pca // For KNN
	
	 make clean // clean up

Run: 
	
	 ./a.out // For KNN
	 cuda-memcheck ./a.out // For debug
