#!/bin/sh

find . -type f -exec sed --in-place 's/g_host->/Provider_GetHost()->/g' '{}' ';'
