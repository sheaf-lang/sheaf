cd ../examples

for TEST in baregpt clevr hydra mlp; do
	cd $TEST
	echo "Running $TEST..."
	if sheaf ./run.shf > /dev/null ; then
	echo "$TEST OK"
	else	echo "$TEST FAIL"
	fi
	cd ..
done
