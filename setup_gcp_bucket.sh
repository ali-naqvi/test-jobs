#!/bin/bash

set -e

# Install gcloud:
#     https://cloud.google.com/sdk/docs/install
# Then following the instructions to:
#     gcloud auth login

# This is the ID of your GCP project.
PROJ_ID=""
# This is the name of the Anyscale cloud from which we would
# like to access GCP buckets.
CLOUD_NAME=""
# Name of the bucket to be created.
BUCKET=""
# Which zone should this bucket be created. E.g., US-WEST2
BUCKET_ZONE=""

echo "Granting Anyscale service account storage access"

LIST_CLOUD=$(anyscale cloud list --name ${CLOUD_NAME} | grep ${CLOUD_NAME})

# Cloud id is the first column. GCP also want it in cld-xxx format,
# instead of the cld_xxx format that Anyscale uses.
CLOUD_ID=$(echo ${LIST_CLOUD} \
	       | sed 's/[ ].*//' \
	       | sed 's/_/-/' \
	       | awk '{print tolower($0)}')

# Also parse out the Anyscale bridge project ID.
BRIDGE_PROJ_ID=$(echo ${LIST_CLOUD} | sed 's/.*\(anyscale-bridge-[a-f0-9]*\).*/\1/g')

# Construct the full service account.
CLOUD_BRIDGE_ACCOUNT="serviceAccount:${CLOUD_ID}@${BRIDGE_PROJ_ID}.iam.gserviceaccount.com"

echo "Cloud account is ${CLOUD_BRIDGE_ACCOUNT}"

# Granting cloud storage admin role.
gcloud projects add-iam-policy-binding ${PROJ_ID} \
       --member=${CLOUD_BRIDGE_ACCOUNT} \
       --role=roles/storage.objectAdmin \
       --user-output-enabled=false

echo "Creating bucket"

# Creating a
gsutil mb -p ${PROJ_ID} -l ${BUCKET_ZONE} gs://${BUCKET}
