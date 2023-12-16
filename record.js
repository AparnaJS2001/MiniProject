let mediaRecorder;
let audioChunks = [];
let audioPlayer = document.getElementById('audioPlayer');

document.getElementById('startRecording').addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);

        audioPlayer.src = audioUrl;
    };

    mediaRecorder.start();

    document.getElementById('startRecording').disabled = true;
    document.getElementById('stopRecording').disabled = false;
});

document.getElementById('stopRecording').addEventListener('click', () => {
    mediaRecorder.stop();

    document.getElementById('startRecording').disabled = false;
    document.getElementById('stopRecording').disabled = true;
    audioChunks = [];
});
