cd /work/rleap1/samuel.stante/genfond/generators
find . -maxdepth 1 -type f | grep -Ev './(_.bash|all.bash)$' | xargs -I {} bash -c '{}'
