#!/usr/bin/env bash
set -euo pipefail

download_file() {
    local url=$1 out=$2
    if command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "${out}" "${url}"
    else
        echo "ERROR: wget or curl is required" >&2
        exit 2
    fi
}

mkdir -p 3rd_party/CARLA_0915

cd 3rd_party/CARLA_0915

download_file https://tiny.carla.org/carla-0-9-15-linux CARLA_0915.tar.gz
tar -xvzf CARLA_0915.tar.gz

# GNU tar 1.35 has a known bug with --keep-newer-files that makes CARLA's
# ImportAssets.sh fail on existing directories in AdditionalMaps_0.9.15.tar.gz.
sed -i 's/tar --keep-newer-files -xvf/tar -xvf/g' ImportAssets.sh

cd Import
download_file https://tiny.carla.org/additional-maps-0-9-15-linux AdditionalMaps_0.9.15.tar.gz
cd ..
bash ImportAssets.sh
