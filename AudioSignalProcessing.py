import librosa
import matplotlib.pyplot as plt
import numpy as np 
from scipy.fft import fft
import math
from tkinter import* 
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import get_window
from tkinter import simpledialog, messagebox

#function to calculate fft
def FFT(input_array): 
    x = np.asarray(input_array, dtype=np.complex128)
    n = len(x)
    if (n & (n - 1)) != 0:
        raise ValueError("Input length must be a power of two")
    
    # Bit-reversal ordering
    def bit_reverse_permute(x):
        n = len(x)
        permuted_x = np.zeros(n, dtype=np.complex128)
        # Calculate bit-reversed indices
        for i in range(n):
            reversed_index = int(f"{i:0{int(np.log2(n))}b}"[::-1], 2)
            permuted_x[reversed_index] = x[i]
        return permuted_x

    x = bit_reverse_permute(x)

    # Start Cooley-Tukey FFT implementation
    m = 2
    while m <= n:
        omega_m = np.exp(-2j * np.pi / m)
        for k in range(0, n, m):
            omega = 1 + 0j
            for j in range(m // 2):
                index = k + j
                t = omega * x[index + m // 2]
                x[index + m // 2] = x[index] - t
                x[index] = x[index] + t
                omega *= omega_m
        m *= 2

    return x

#calculate stft
def STFT(samples, sampling_rate, n_fft=2048, hop_length=512, window='hann'):
    # Calculate the window length and number of windows
    window_length = n_fft
    num_windows = (len(samples) - window_length) // hop_length + 1    
    window_function = get_window(window, window_length)
    stft_result = np.empty((n_fft // 2 + 1, num_windows), dtype=np.complex64)
    for i in range(num_windows):
        start_index = i * hop_length
        end_index = start_index + window_length
        segment = samples[start_index:end_index] * window_function       
        # Compute FFT and store only positive frequencies (half of the FFT result)
        stft_result[:, i] = np.fft.rfft(segment, n=n_fft)        
    return stft_result

#padding the signal 
def zero_pad_to_power_of_two(audio):
    """Zero-pad the input array to the next power of two."""
    length = len(audio)
    # Calculate the next power of two
    next_power_of_two = 1
    while next_power_of_two < length:
        next_power_of_two *= 2
    
    # Pad the array with zeros to the next power of two
    padded_audio = np.pad(audio, (0, next_power_of_two - length), mode='constant')
    
    return padded_audio


#time domain plot   
def time_domain_plot(samples, sampling_rate):
    # Calculate the time axis
    time = np.arange(len(samples)) / sampling_rate
    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    return plt


#fft plot
def fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
   # yf = fft(audio) -built in
    padded_audio = zero_pad_to_power_of_two(audio)
    yf=FFT(padded_audio)  #manual
    print(yf)   
    xf = np. linspace(0.0, 1.0/ (2.0*T),int(n/2))
    plt.figure()
    plt. plot(xf, 2.0/n * np.abs (yf [:n//2])) 
    plt. xlabel ("Frequency")
    plt.ylabel ( "Magnitude")
    plt.grid()    
    return plt


#spectrogram plot
def spectrogram(samples,sampling_rate):
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(STFT(samples,sampling_rate)), ref=np.max),
                         sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    return plt



#pitch detection
def pitch_detection(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    padded_audio = zero_pad_to_power_of_two(audio)
    yf=FFT(padded_audio)
    xf = np. linspace(0.0, 1.0/ (2.0*T),int(n*10))
    max_index = np.argmax(np.abs(yf[:n//2]))
    pitch = xf[max_index]
    plt.figure()
    plt.plot([0, sampling_rate/2], [pitch, pitch], 'r--', linewidth=2)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Pitch Detection")
    plt.grid()
    return plt


#spectrgram harmonics
def spectrogram_harmonics(samples, sampling_rate):
    # Compute the Short-Time Fourier Transform (STFT)
    #D = librosa.stft(samples)-build in
    D=STFT(samples,sampling_rate) #manual
    S = np.abs(D)**2
    # Compute the minimum and maximum frequencies
    n_fft = D.shape[1]
    fmin = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)[1]
    fmax = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)[-1]

    # Compute the harmonic components using the librosa.display.specshow_components function
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                         sr=sampling_rate, hop_length=512,
                                         x_axis='time', y_axis='log',
                                         cmap='inferno',
                                         )

    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt

#audio compression
def audio_compression(audio, sampling_rate, compression_ratio):
    
    compression_ratio = simpledialog.askfloat("Compression Input", "Enter Compression Ratio:", minvalue=0.1, maxvalue=100.0)
    # Divide the audio signal into frames
    frame_size = int(sampling_rate * 0.02)  # 20ms frame size
    frame_stride = int(sampling_rate * 0.01)  # 10ms frame stride
    n_frames = math.ceil(len(audio) / frame_stride)
    frames = [audio[i*frame_stride:i*frame_stride+frame_size] for i in range(n_frames)]

    # Compute the FFT for each frame
    fft_frames = [np.fft.fft(frame) for frame in frames]

    # Apply compression by keeping only the most significant coefficients
    compression_factor = int(len(fft_frames[0]) / compression_ratio)
    compressed_fft_frames = [fft_frame[:compression_factor] for fft_frame in fft_frames]
    # Reconstruct the audio signal from the compressed FFT coefficients
    reconstructed_frames = [np.fft.ifft(fft_frame) for fft_frame in compressed_fft_frames]
    reconstructed_audio = np.hstack(reconstructed_frames)

    # Normalize the reconstructed audio signal
    reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))
    np.seterr(all='ignore')
    reconstructed_audio = reconstructed_audio.astype(np.float32)

    # Plot the original and compressed audio signals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, len(audio)/sampling_rate, len(audio)), audio)
    plt.title('Original Audio')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, len(reconstructed_audio)/sampling_rate, len(reconstructed_audio)), reconstructed_audio)
    plt.title('Compressed Audio')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt


#power spectral density
def Power_Spectral_Density(samples, sampling_rate):
    """This gives an idea of the distribution of energy across different frequencies."""    
    n = len(samples)
    nfft = 2**np.ceil(np.log2(n)).astype(int)  # Round up to the nearest power of 2
    f = np.fft.fftfreq(nfft, 1/sampling_rate)
    xf = np.fft.fft(samples, nfft)
    psd = np.abs(xf)**2 / nfft

    plt.figure()
    plt.plot(f, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.xlim(-1000,1000)
    return plt

song = "song.wav"
samples,sampling_rate=librosa.load(song,sr=None,mono=True,offset=0.0,duration=None)
n=len(samples)
print(n)
'''time_domain_plot(samples, sampling_rate)
fft_plot(samples,sampling_rate)
spectrogram(samples,sampling_rate)
pitch_detection(samples,sampling_rate)
spectrogram_harmonics(samples, sampling_rate)
audio_compression(samples, sampling_rate, 60)
Power_Spectral_Density(samples, sampling_rate)'''

# tKinter Part

root = Tk()
root.title("Audio Signal Processing using FFT")
root.geometry("800x800")
root.configure(bg="lightblue")

options = [ "Time Domain Plot",
           "FFT Plot",
           "Spectogram",
           "Pitch Detection",
           "Spectrogram Harmonics",
           "Audio Compression",
           "Power Spectral density"]

#Drop down boxes
# Function to display selected option
def show():
    global plot_canvas  # Declare plot_canvas as global

    # Clear previously displayed plot if it exists
    if 'plot_canvas' in globals():
        plot_canvas.get_tk_widget().destroy()

    selected_option = clicked.get()
    if selected_option == "Time Domain Plot":
        plot1 = time_domain_plot(samples, sampling_rate)
    elif selected_option == "FFT Plot":
        plot1 = fft_plot(samples, sampling_rate)
    elif selected_option == "Spectogram":
        plot1 = spectrogram(samples,sampling_rate)
    elif selected_option == "Pitch Detection":
        plot1 = pitch_detection(samples,sampling_rate)
    elif selected_option == "Spectrogram Harmonics":
        plot1 = spectrogram_harmonics(samples, sampling_rate)
    elif selected_option == "Audio Compression":
        plot1= audio_compression(samples, sampling_rate, 60)
    elif selected_option == "Power Spectral density":
        plot1 = Power_Spectral_Density(samples, sampling_rate)
    else:
        plot1 = "Invalid Option"

    # Get the current figure associated with the plot
    fig = plot1.gcf()

    # Embed plot in Tkinter window
    plot_canvas = FigureCanvasTkAgg(fig, master=root)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack()

# Dropdown menu
clicked = StringVar()
clicked.set("Choose a Signal processing technique")  # Set default option
drop = OptionMenu(root, clicked, *options)
drop.configure(bg="white",font=("Arial", 12, "italic"),relief="ridge")
drop.pack(pady=20)

# Button to trigger display
myButton = Button(root, text="Show Plot", command=show,bg="green", 
    fg="white", 
    font=("Arial", 12, "bold"),  
    relief="raised",
    bd=2, )
myButton.pack(pady=30)

def on_closing():
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()