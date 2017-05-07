#!/bin/bash
make
for var1 in 1 2 3 4 5 
do
	var2=train.txt
	vartest=test.txt
	var3=${var1}${var2}
	./svm-train -t 0 $var3
	./svm-predict ${var1}${vartest} ${var3}".model" output >> MQ2007.log
done
