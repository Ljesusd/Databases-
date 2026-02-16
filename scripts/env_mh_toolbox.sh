#!/usr/bin/env bash
# Ensure ezc3d can locate its dylib on macOS
export DYLD_LIBRARY_PATH="/Users/lv/Library/Python/3.14/lib/python/site-packages/ezc3d${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
