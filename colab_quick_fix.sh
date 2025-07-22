#!/bin/bash
echo "🔧 Quick OpenFHE Library Fix"
echo "============================="

# Check if OpenFHE libraries exist but aren't linked properly
if [ -f "/usr/local/lib/libOPENFHEcore.so" ]; then
    echo "✅ OpenFHE core library found"
    
    # Update library cache
    echo "🔄 Updating library cache..."
    sudo ldconfig
    
    # Verify library linking
    echo "🔍 Checking library dependencies..."
    ldd /usr/local/lib/libOPENFHEcore.so | head -3
    
    echo "✅ Library cache updated - try building again"
else
    echo "❌ OpenFHE libraries not found - need to install first"
fi 