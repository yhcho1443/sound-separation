#!/bin/bash
# Fast downloader for FUSS dataset into Colab local storage (/content/fuss_data)
# Saves disk space by deleting the .tar.gz archive immediately after extraction.
set -e

FUSS_ROOT_DIR="${FUSS_ROOT_DIR:-/content/fuss_data}"
INCLUDE_REVERB="${INCLUDE_REVERB:-0}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --root-dir)
      FUSS_ROOT_DIR="$2"
      shift 2
      ;;
    --include-reverb)
      INCLUDE_REVERB=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

DOWNLOAD_DIR="${FUSS_ROOT_DIR}/download"
DEV_DATA_DIR="${FUSS_ROOT_DIR}/fuss_dev"
SS_DIR="${DEV_DATA_DIR}/ssdata"
SS_REVERB_DIR="${DEV_DATA_DIR}/ssdata_reverb"

SSDATA_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata.tar.gz"
SSDATA_REVERB_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata_reverb.tar.gz"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${DEV_DATA_DIR}"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v aria2c >/dev/null 2>&1; then
    echo "Downloading with aria2c (multi-connection): ${url}"
    aria2c -x 8 -s 8 -k 1M -d "$(dirname "${dest}")" -o "$(basename "${dest}")" "${url}"
  else
    echo "Downloading with curl: ${url}"
    curl -L --output "${dest}" "${url}"
  fi
}

echo "============================================================"
echo "Target FUSS local directory: ${DEV_DATA_DIR}"
echo "============================================================"

if [ ! -d "${SS_DIR}" ]; then
  if [ ! -s "${DOWNLOAD_DIR}/ssdata.tar.gz" ]; then
    download_file "${SSDATA_URL}" "${DOWNLOAD_DIR}/ssdata.tar.gz"
  fi
  echo "Extracting ssdata.tar.gz to ${DEV_DATA_DIR}..."
  tar xzf "${DOWNLOAD_DIR}/ssdata.tar.gz" -C "${DEV_DATA_DIR}"
  echo "Removing archive ${DOWNLOAD_DIR}/ssdata.tar.gz to free Colab disk space..."
  rm -f "${DOWNLOAD_DIR}/ssdata.tar.gz"
else
  echo "${SS_DIR} already exists, skipping download."
fi

if [ "${INCLUDE_REVERB}" == "1" ]; then
  if [ ! -d "${SS_REVERB_DIR}" ]; then
    if [ ! -s "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz" ]; then
      download_file "${SSDATA_REVERB_URL}" "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz"
    fi
    echo "Extracting ssdata_reverb.tar.gz to ${DEV_DATA_DIR}..."
    tar xzf "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz" -C "${DEV_DATA_DIR}"
    echo "Removing archive ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz to free Colab disk space..."
    rm -f "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz"
  else
    echo "${SS_REVERB_DIR} already exists, skipping download."
  fi
fi

echo "============================================================"
echo "FUSS dataset is ready at: ${SS_DIR}"
df -h "${FUSS_ROOT_DIR}"
echo "============================================================"
