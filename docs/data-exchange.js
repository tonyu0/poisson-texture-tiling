const fileInput = document.getElementById('fileInput');
const enableProcessCheckBox = document.getElementById('enableProcessCheckBox');
const uploadForm = document.getElementById('uploadForm');
const imagePreviewCanvas = document.getElementById('imagePreview');
const ctx = imagePreviewCanvas.getContext('2d');
const status = document.getElementById('status');

createModule().then(Module => {
    instance = Module;
    status.innerHTML += "- WASM loaded!<br>";
});

const updateStatus = (text) => {
    status.innerHTML += `${text}<br>`;
    // リアルタイムでステータスを更新する。
    // そのままだと画像処理が終わるまでDOMが書き変わらないので、処理をasyncとして、awaitでDOM更新を挟む。
    // 1ミリ秒だけ処理を譲ることで描画を許可する。
    return new Promise(resolve => setTimeout(resolve, 1));
};

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Init 
    ctx.clearRect(0, 0, imagePreviewCanvas.width, imagePreviewCanvas.height);
    status.innerHTML = ""

    // const selectedFile = e.target.files[0];
    const selectedFile = fileInput.files[0];

    if (selectedFile) {

        await updateStatus("- File selected: " + selectedFile.name);
        const reader = new FileReader();

        reader.onload = async (event) => {
            await updateStatus("- FileReader loaded");
            const img = new Image();
            img.onload = async () => {
                await updateStatus("- Input image size: <b>" + String(img.width) + "</b>px x <b>" + String(img.height) + "</b>px");
                await updateStatus("- N (pixel count) = <b>" + String(img.width * img.height) + "</b>.");
                // console.log(img.width, img.height)
                imagePreviewCanvas.width = img.width;
                imagePreviewCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                // Fetch pixel data from Canvas which shows the specified image
                const imageData = ctx.getImageData(0, 0, imagePreviewCanvas.width, imagePreviewCanvas.height);

                // Main process of WASM C++
                const calcType = document.getElementById("enableFFT").checked ? 0 : 1;
                await updateStatus("- Selected calculation method is: " + ["FFT", "Direct method using Cholesky decomposition"][calcType])

                const startTime = performance.now()
                await handleImageWithWasm(imageData, calcType);
                await updateStatus("- Main process took: <b>" + String(Math.round((performance.now() - startTime) * 100) / 100) + "</b> ms");

                // Canvas expects RGBA
            };
            img.src = event.target.result;
        };
        // Read the file as a Data URL (Base64 encoded string)
        reader.readAsDataURL(selectedFile);
    }
});

async function handleImageWithWasm(imageData, calcType) {
    // Overview: 
    // ImageData.data(Uint8Array) -> pBuffer (memory "start pointer" allocated on the WASM C++ side (uint*)) -> 
    // Copy ImageData.data to pBuffer -> [process with WASM C++] -> Copy a result to new ImageData.data

    const width = imageData.width;
    const height = imageData.height;

    // 1. Allocate buffer memory from C++ 
    await updateStatus("- Allocate enough buffer within WASM");
    const pBuffer = instance._allocate_buffer(width, height);

    // 2. Map imageData.data (Uint8Array) to memory location pointed to by pBuffer (instance.HEAPU8 is all the memory space of WASM)
    await updateStatus("- Set data from ImageData of Canvas to the buffer within WASM");
    instance.HEAPU8.set(imageData.data, pBuffer);

    // 3. Run main process (poisson texture tiling) of WASM C++
    if (enableProcessCheckBox.checked) {
        await updateStatus("- Running main process (solving the Poisson Equation)");
        instance._process_image(pBuffer, width, height, calcType);
    }

    // 4. Read the resulted image from buffer within WASM (the required size is width * height * RGBA channels)
    const resultView = new Uint8Array(instance.HEAPU8.buffer, pBuffer, width * height * 4);

    // Copy processed data to Canvas
    await updateStatus("- Copying the processed image to Canvas");
    imageData.data.set(resultView);
    ctx.putImageData(imageData, 0, 0);

    { // Demo tiling 3x3
        // パターン用のCanvasを裏で作り、createPatternでそれを敷き詰める

        // Canvas for pattern
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        const tempImageData = tempCtx.createImageData(width, height);
        tempImageData.data.set(resultView);
        tempCtx.putImageData(tempImageData, 0, 0);

        // Canvas for display
        const displayCanvas = document.getElementById('displayCanvas');
        displayCanvas.width = width * 3;
        displayCanvas.height = height * 3;
        const displayCtx = displayCanvas.getContext('2d');

        // createPattern using tempCanvas
        const pattern = displayCtx.createPattern(tempCanvas, 'repeat');

        if (pattern) {
            displayCtx.fillStyle = pattern;
            displayCtx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);
        }
    }

    // Free buffer memory from C++
    await updateStatus("- Freeing allocated memory within WASM");
    instance._free_buffer(pBuffer);

    await updateStatus("- Image Processed and Rendered!<br>");
}