#!/bin/bash
# Build Squirrel Finder as a double-clickable .app and launch it.
set -euo pipefail
cd "$(dirname "$0")"
swift build -c release
APP=build/SquirrelFinder.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/SquirrelFinder "$APP/Contents/MacOS/SquirrelFinder"
cp -R Assets "$APP/Contents/Resources/Assets"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Squirrel Finder</string>
  <key>CFBundleIdentifier</key><string>dev.local.squirrelfinder</string>
  <key>CFBundleExecutable</key><string>SquirrelFinder</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
open "$APP"
