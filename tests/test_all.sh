for d in */ ; do
    echo "$d"
    cd $d
    igm_run
    cd ..
done

