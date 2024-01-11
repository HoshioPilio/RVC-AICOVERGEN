# RVC-AICOVERGEN
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: https://www.youtube.com/watch?v=2qZuE4WM7CM

Setup Guide: https://www.youtube.com/watch?v=pdlhk4vVHQk
hugging face space

![](data:image/png;base64,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)[https://huggingface.co/spaces/Lask8/AICoverGen-v2]

WebUI is under constant development and testing, but you can try it out right now on both local and colab!

## Changelog

- add new theme and title little bit

- WebUI for easier conversions and downloading of voice models
- Support for cover generations from a local audio file
- Option to keep intermediate files generated. e.g. Isolated vocals/instrumentals
- Download suggested public voice models from table with search/tag filters
- Support for Pixeldrain download links for voice models
- Implement new rmvpe pitch extraction technique for faster and higher quality vocal conversions
- Volume control for AI main vocals, backup vocals and instrumentals
- Index Rate for Voice conversion
- Reverb Control for AI main vocals
- Local network sharing option for webui
- Extra RVC options - filter_radius, rms_mix_rate, protect
- Local file upload via file browser option
- Upload of locally trained RVC v2 models via WebUI
- Pitch detection method control, e.g. rmvpe/mangio-crepe
- Pitch change for vocals and instrumentals together. Same effect as changing key of song in Karaoke.
- Audio output format option: wav or mp3.

## Update AICoverGen to latest version

Install and pull any new requirements and changes by opening a command line window in the `AICoverGen` directory and running the following commands.

```
pip install -r requirements.txt
git pull
```

For colab users, simply click `Runtime` in the top navigation bar of the colab notebook and `Disconnect and delete runtime` in the dropdown menu. 
Then follow the instructions in the notebook to run the webui.

## Colab notebook

For those without a powerful enough NVIDIA GPU, you may try AICoverGen out using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laynz28/RCV-AI-COVER-ALL/blob/main/AICoverGen_V2.ipynb)

For those who face issues with Google Colab notebook disconnecting after a few minutes, here's an alternative that doesn't use the WebUI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ardha27/AICoverGen-NoUI-Colab/blob/main/CoverGen_No_UI.ipynb)

For those who want to run this locally, follow the setup guide below.

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Also follow this [guide](https://realpython.com/installing-python/) to install Python **VERSION 3.9** if you haven't already. Using other versions of Python may result in dependency conflicts.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Install sox

Follow the instructions [here](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/) to install sox and add it to your Windows path environment.

### Clone AICoverGen repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/HoshioPilio/RVC-AICOVERGEN
cd RVC-AICOVERGEN
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python src/download_models.py
```


## Usage with WebUI

To run the AICoverGen WebUI, run the following command.

```
python src/webui.py
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--share`                                  | Create a public URL. This is useful for running the web UI on Google Colab. |
| `--listen`                                 | Make the web UI reachable from your local network. |
| `--listen-host LISTEN_HOST`                | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`                | The listening port that the server will use. |

Once the following output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the WebUI.

### Download RVC models via WebUI

![](images/Tadownload_models.png?raw=true)

Navigate to the `Download model` tab, and paste the download link to the RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download. You may refer to the examples for how the download link should look like.
The downloaded zip file should contain the .pth model file and an optional .index file.

Once the 2 input fields are filled in, simply click `Download`! Once the output message says `[NAME] Model successfully downloaded!`, you should be able to use it in the `Generate` tab after clicking the refresh models button!


### Running the pipeline via WebUI

![](images/webui_generate.png?raw=true)

- From the Voice Models dropdown menu, select the voice model to use. Click `Update` if you added the files manually to the [rvc_models](rvc_models) directory to refresh the list.
- In the song input field, copy and paste the link to any song on YouTube or the full path to a local audio file.
- Pitch should be set to either -12, 0, or 12 depending on the original vocals and the RVC AI modal. This ensures the voice is not *out of tune*.
- Other advanced options for Voice conversion and audio mixing can be viewed by clicking the accordion arrow to expand.

Once all Main Options are filled in, click `Generate` and the AI generated cover should appear in a less than a few minutes depending on your GPU.

## Usage with CLI

### Manual Download of RVC models

Unzip (if needed) and transfer the `.pth` and `.index` files to a new folder in the [rvc_models](rvc_models) directory. Each folder should only contain one `.pth` and one `.index` file.

The directory structure should look something like this:
```
├── rvc_models
│   ├── John
│   │   ├── JohnV2.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── May
│   │   ├── May.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── MODELS.txt
│   └── hubert_base.pt
├── mdxnet_models
├── song_output
└── src
 ```

### Running the pipeline

To run the AI cover generation pipeline using the command line, run the following command.

```
python src/main.py [-h] -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [-k | --keep-files | --no-keep-files] [-ir INDEX_RATE] [-fr FILTER_RADIUS] [-rms RMS_MIX_RATE] [-palgo PITCH_DETECTION_ALGO] [-hop CREPE_HOP_LENGTH] [-pro PROTECT] [-mv MAIN_VOL] [-bv BACKUP_VOL] [-iv INST_VOL] [-pall PITCH_CHANGE_ALL] [-rsize REVERB_SIZE] [-rwet REVERB_WETNESS] [-rdry REVERB_DRYNESS] [-rdamp REVERB_DAMPING] [-oformat OUTPUT_FORMAT]
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `-i SONG_INPUT`                            | Link to a song on YouTube or path to a local audio file. Should be enclosed in double quotes for Windows and single quotes for Unix-like systems. |
| `-dir MODEL_DIR_NAME`                      | Name of folder in [rvc_models](rvc_models) directory containing your `.pth` and `.index` files for a specific voice. |
| `-p PITCH_CHANGE`                          | Change pitch of AI vocals in octaves. Set to 0 for no change. Generally, use 1 for male to female conversions and -1 for vice-versa. |
| `-k`                                       | Optional. Can be added to keep all intermediate audio files generated. e.g. Isolated AI vocals/instrumentals. Leave out to save space. |
| `-ir INDEX_RATE`                           | Optional. Default 0.5. Control how much of the AI's accent to leave in the vocals. 0 <= INDEX_RATE <= 1. |
| `-fr FILTER_RADIUS`                        | Optional. Default 3. If >=3: apply median filtering median filtering to the harvested pitch results. 0 <= FILTER_RADIUS <= 7. |
| `-rms RMS_MIX_RATE`                        | Optional. Default 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1). 0 <= RMS_MIX_RATE <= 1. |
| `-palgo PITCH_DETECTION_ALGO`              | Optional. Default rmvpe. Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals). |
| `-hop CREPE_HOP_LENGTH`                    | Optional. Default 128. Controls how often it checks for pitch changes in milliseconds when using mangio-crepe algo specifically. Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy. |
| `-pro PROTECT`                             | Optional. Default 0.33. Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable. 0 <= PROTECT <= 0.5. |
| `-mv MAIN_VOCALS_VOLUME_CHANGE`            | Optional. Default 0. Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels. |
| `-bv BACKUP_VOCALS_VOLUME_CHANGE`          | Optional. Default 0. Control volume of backup AI vocals. |
| `-iv INSTRUMENTAL_VOLUME_CHANGE`           | Optional. Default 0. Control volume of the background music/instrumentals. |
| `-pall PITCH_CHANGE_ALL`                   | Optional. Default 0. Change pitch/key of background music, backup vocals and AI vocals in semitones. Reduces sound quality slightly. |
| `-rsize REVERB_SIZE`                       | Optional. Default 0.15. The larger the room, the longer the reverb time. 0 <= REVERB_SIZE <= 1. |
| `-rwet REVERB_WETNESS`                     | Optional. Default 0.2. Level of AI vocals with reverb. 0 <= REVERB_WETNESS <= 1. |
| `-rdry REVERB_DRYNESS`                     | Optional. Default 0.8. Level of AI vocals without reverb. 0 <= REVERB_DRYNESS <= 1. |
| `-rdamp REVERB_DAMPING`                    | Optional. Default 0.7. Absorption of high frequencies in the reverb. 0 <= REVERB_DAMPING <= 1. |
| `-oformat OUTPUT_FORMAT`                   | Optional. Default mp3. wav for best quality and large file size, mp3 for decent quality and small file size. |


## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
