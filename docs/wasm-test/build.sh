emcc main.cpp -o index.js --bind -s MODULARIZE=1 -s EXPORT_NAME='createModule'

# when check locally
# python3 -m http.server