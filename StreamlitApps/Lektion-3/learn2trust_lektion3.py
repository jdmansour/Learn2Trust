import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import pandas as pd
import random
import torchvision.transforms as T
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix

import streamlit as st


# Caching
@st.cache_data()
def load_data():
    Thorax_img = Image.open('StreamlitApps/Lektion-3/l2t_data/person1_bacteria_2.jpeg')
    Thorax_img = T.Resize(size=512)(Thorax_img)
    Thorax_img = T.CenterCrop(size=[512, 512])(Thorax_img)

    Skin_img = Image.open('StreamlitApps/Lektion-3/l2t_data/ISIC_0015719.jpg')
    Skin_img = T.Resize(size=512)(Skin_img)
    Skin_img = T.CenterCrop(size=[512, 512])(Skin_img)

    Brain_img = Image.open('StreamlitApps/Lektion-3/l2t_data/OASIS0001.png')
    Brain_img = T.Resize(size=512)(Brain_img)
    Brain_img = T.CenterCrop(size=[512, 512])(Brain_img)

    return Thorax_img, Skin_img, Brain_img


#############
# Functions #
#############
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

def concat(img1, img2, img3):
    cat = Image.new(mode='RGB', size=(img1.width*3+100, img1.height), color=(255, 255, 255))
    cat.paste(img1, (0, 0))
    cat.paste(img2, (img1.width+50, 0))
    cat.paste(img3, (img1.width*2+100, 0))
    return cat


################
# Introduction #
################
def intro(session_state):
    st.write("""
    ## 1. Einführung
    """)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("""In dieser Lektion geht es um Bilddatensätze im Zusammenhang mit Künstlicher Intelligenz.""")
        st.write("""Datensätze bilden die Grundlage bei der Entwicklung von KI. Die Qualität der Daten, die für Training, Validierung und Test einer KI verwendet werden, ist entscheidend dafür, ob die KI zur Analyse unbekannter Daten eingesetzt werden kann. Grundsätzlich gilt, dass ein Datensatz möglichst **umfangreich** sein sollte. Je umfangreicher ein Datensatz ist, desto besser kann eine KI trainiert werden. """)
        st.write("""Nach der Datenerfassung erfolgen mehrere Schritte, welche den Datensatz modifizieren und für die Trainingsphase vorbereiten. Der Datensatz wird zunächst bei der Datenauswahl und -aufteilung so zusammengestellt, dass er zur Aufgabe, für welche die KI später eingesetzt werden soll, passt. Dabei sollten die Daten möglichst **vielfältig** ausgewählt werden, sodass die KI später bessere Vorhersagen für unbekannte Daten treffen kann. Diese Vielfältigkeit sollte auch bei der Aufteilung in Trainings-, Validierungs- und Testdatensatz eingehalten werden. """)
        
        st.write("""Je nachdem, ob es sich um ein überwachtes oder unüberwachtes Lernverfahren handelt, werden die Daten annotiert. Hierbei ist für die Datensatzqualität entscheidend, dass **verlässlich** annotiert wird, denn eine KI kann nur verlässlich angewandt werden, wenn sie mit korrekten Eingabe-Grundwahrheit-Paaren traininert wurde.  Bevor die Daten zum Trainings der KI verwendet werden, werden sie außerdem vorverarbeitet und gegebenenfalls augmentiert, um die Anzahl an Trainingsdaten zu erhöhen. """)
        st.write("""In den Unterkapiteln dieser Lektion werden Annotation, Vorverarbeitung und Augmentierung von 
        Bilddatensätzen genauer erklärt und demonstriert. Außerdem wird in dieser Lektion gezeigt, wie evaluiert werden kann, wie gut eine KI in der Testphase Vorhersagen für Daten des Testdatensatzes trifft.""")
        
    with col2:
        st.image('StreamlitApps/Lektion-3/l2t_images/datenpipeline.png', caption='Pipeline von der Datenerfassung bis zum Training des KI-Modells.')

    st.markdown("---")
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.write("""
        ### Unterkapitel
        Auswahl über Seitenleiste

        1. Einführung
        2. Annotationen
        3. Vorverarbeitung
        4. Augmentierung
        5. Evaluationsmetriken

        """)

    st.markdown("---")


###############
# Annotations #
###############
def annotations(session_state):
    st.write("""
    ## 2. Annotationen
    """)

    st.write("""Annotationen bei Entwicklung einer KI für die medizinische Bildanalyse sind **Zuordnungen von Informationen zu einem Bild**. Für eine Klassifikationsaufgabe muss der Bilddatensatz gelabelt werden. Für eine Segmentierungsaufgabe wird das Bild in zusammenhängende Bildbereiche eingeteilt, um z.B. anatomisch zusammenhängende oder diagnostisch oder therapeutisch relevante Bildbereiche abzugrenzen. """)
    st.markdown("---")

    st.write("""
        ### Label
        """)
    st.write("""Beim Labeln wird jedem Bild eine Klasse aus einer vordefinierten Auswahl an Klassen zugewiesen, z.B. 'Pneumonie' oder 'keine Pneumonie'.""")

    col1, col2, col3, col4, col5 = st.columns([2,2,0.5,2,2])
    with col1:
        st.image('StreamlitApps/Lektion-3/l2t_images/pneumonielabel.png')  # , caption='Label')
    with col2:
        st.write("""**Label**: """)
        st.write(""" 'Pneumonie' """)
    with col4:
        st.image('StreamlitApps/Lektion-3/l2t_images/keinepneumonielabel.png')  # , caption='Label')
    with col5:
        st.write("""**Label**: """)
        st.write(""" 'keine Pneumonie' """)

    st.markdown("---")

    st.write("""
            ### Segmentierung
            """)
    st.write("""Bei der Segmentierung werden Pixel einer inhaltlich zusammenhängenden Region zusammengefasst. Werden den ermittelten Bildbereichen zusätzlich Label zugewiesen, dann handelt es sich um semantische Segmentierung. """)

    col1, col2, col3, col4, col5 = st.columns([1, 2, 0.2, 2,1])

    with col2:
        st.image('StreamlitApps/Lektion-3/l2t_images/segmentierungsbeispiel.png')
    with col3:
        st.write('')
        st.write('')
        st.write('')
        st.image('StreamlitApps/Lektion-3/l2t_images/blau.png', width=25)
        st.image('StreamlitApps/Lektion-3/l2t_images/braun.png', width=25)
        st.image('StreamlitApps/Lektion-3/l2t_images/rosa.png', width=25)
        st.image('StreamlitApps/Lektion-3/l2t_images/rot.png', width=25)
        st.image('StreamlitApps/Lektion-3/l2t_images/gruen.png', width=25)
    with col4:
        st.write("""**Label für semantische Segmentierung**: """)
        st.write('  rechter Lungenflügel')
        st.write('  linker Lungenflügel')
        st.write('  rechtes Schlüsseslbein')
        st.write('  linkes Schlüsselbein')
        st.write('  Herz')
        

#################
# Preprocessing #
#################
def preprocessing(session_state):
    st.write("""
    ## 3. Vorverarbeitung
    """)
    st.write("""Im Rahmen der Vorverarbeitung werden die Bilder bezüglich ihrer Bildbereiche, Größe, Auflösung und Wertebereich **vereinheitlicht.**""")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        st.image('StreamlitApps/Lektion-3/l2t_images/Vorverarbeitung.png', caption='Vorverarbeitung')
    with col3:
        st.write("")

    st.markdown("---")

    st.write("""### Methoden der Vorverarbeitung""")
    st.write("""Häufige Methoden der Vorverarbeitung sind:
                """)

    with st.expander("1. Resizing"):
        st.write("Größenänderung: Änderung der Bilddimensionen und -auflösung")
    with st.expander("2. Cropping"):
        st.write("Zuschneiden des Bildbereiches")
    with st.expander("3. Padding"):
        st.write("Auffüllen des Bildes außerhalb der Bildränder, wenn eine bestimmte Bildgröße erhalten werden soll")

    st.markdown("---")

    Thorax_img, Skin_img, Brain_img = load_data()
    cat_orig_imgs = concat(Thorax_img, Skin_img, Brain_img)

    Methoden = ['Resizing', 'Cropping', 'Padding']
    methods = st.multiselect('Wähle beliebige Methoden zum Ausprobieren aus: ', Methoden, [])

    preproc_img = Thorax_img
    preproc_img2 = Skin_img
    preproc_img3 = Brain_img

    preprocess = []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        # '''
        # # QUESTION:
        # - Check if slider has potential scaling to have 2^n steps sizes
        # - And img size in both axis or just rectangular?
        # '''
        if "Resizing" in methods:
            st.write("""
            ##### Resizing
            """)
            size = st.slider('Bildgröße',64, 512, 512)
            interpolation_mode = st.radio("Interpolationsmethode: ", ('nearest', 'bilinear', 'bicubic'))
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            if interpolation_mode == 'nearest': interpolation = Image.NEAREST
            if interpolation_mode == 'bilinear': interpolation = Image.BILINEAR
            if interpolation_mode == 'bicubic': interpolation = Image.BICUBIC
            preprocess.append(T.Resize(size=(size,size), interpolation=interpolation))


    with col2:
        # '''
        # # QUESTION:
        # CenterCrop for simplicity or Crop with 4 input parameters?
        # '''
        if "Cropping" in methods:
            st.write("""
            ##### Cropping
            """)
            output_size = st.slider('Bildgröße in Pixel', 128, 512, 512)
            preprocess.append(T.CenterCrop(size=output_size))


    with col3:
        if 'Padding' in methods:
            st.write("""
            ##### Padding
            """)
            strength_padding = st.slider('Anzahl Pixel für Padding an jeden Bildrand',0, 100, 15)
            mode = st.radio(label = "Paddingmodus: ", options=['constant', 'edge', 'reflect', 'symmetric'])
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            preprocess.append(T.Pad(padding=strength_padding, padding_mode=mode))

    st.markdown("---")

    if methods:
        preprocess = torch.nn.Sequential(*preprocess)
        preproc_img = preprocess(preproc_img)
        preproc_img2 = preprocess(preproc_img2)
        preproc_img3 = preprocess(preproc_img3)
        cat_preproc_imgs = concat(preproc_img, preproc_img2, preproc_img3)

        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 3, 1])
        with col1:
            st.write("")
        with col2:
            fig, ax = plt.subplots(2)
            ax[0].imshow(cat_orig_imgs, cmap='gray')
            ax[0].axis('off')
            ax[1].axis('off')
            st.pyplot(fig)
        with col3:
            st.image('StreamlitApps/Lektion-3/l2t_images/Pfeil.png', caption='Vorverarbeitung')
        with col4:
            fig3, ax3 = plt.subplots(2)
            ax3[0].imshow(cat_preproc_imgs, cmap='gray')
            ax3[0].axis('off')
            ax3[1].axis('off')
            st.pyplot(fig3)
        with col5:
            st.write("")


def augmentation(session_state):
    st.write("""
    ## 4. Augmentierung
    """)
    st.write("""Augmentierung stellt eine Erweiterung des Datensatzes durch Hinzufügen von zusätzlichen Daten dar. Das Ziel dabei ist, dass der Datensatz größer und vielseitiger wird. Die Daten, die bei der Augmentierung entstehen, können modifizierte
    Kopien vorhandener Daten oder synthetische Daten sein.""")

    st.markdown("---")

    st.write("""### Methoden der Augmentierung""")

    with st.expander("1. Affine Transformationen"):
        st.write("Rotation, Translation, Skalierung, Scherung")
    with st.expander("2. Farbjitter"):
        st.write("Zufällige Veränderungen von Helligkeit, Sättigung und Kontrast")
    with st.expander("3. Random Cropping"):
        st.write("Zufälliges Ausschneiden von Bildbereichen")
    with st.expander("4. Gaußscher Unschärfefilter"):
        st.write("Weichzeichnung")

    st.markdown("---")

    Thorax_img, Skin_img, Brain_img = load_data()
    cat_orig_imgs = concat(Thorax_img, Skin_img, Brain_img)

    Augmentierungsmethoden = ['Affine Transformationen', 'Gaußscher Unschärfefilter', 'Farbjitter', 'Zufälliges Zuschneiden', 'Spiegeln', 'Rauschen']  # 'Zufällige affine Transformationen',
    methods = st.multiselect('Methoden: ', Augmentierungsmethoden, [])

    aug_img = Thorax_img
    aug_img2 = Skin_img
    aug_img3 = Brain_img

    augmentation = []

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

    # '''
    # Affine without randomness
    # '''
    with col1:
        if "Affine Transformationen" in methods:
            st.write("""
            ### Affine Transformationen
            """)
            angle = st.slider('Rotation in Grad [-180, 180]', -180, 180, 0)
            t_x = st.slider('Translation in X-Richtung', 0, 256, 0)
            t_y = st.slider('Translation in Y-Richtung', 0, 256, 0)
            scale = st.slider('Skalierung', 0.1, 1.0, 1.0)
            shear = st.slider('Scherung in Grad [0, 180]', 0, 180, 0)
            aug_img = T.functional.affine(img=aug_img, angle=angle, translate=(t_x, t_y), scale=scale, shear=shear)
            aug_img2 = T.functional.affine(img=aug_img2, angle=angle, translate=(t_x, t_y), scale=scale, shear=shear)
            aug_img3 = T.functional.affine(img=aug_img3, angle=angle, translate=(t_x, t_y), scale=scale, shear=shear)

    with col2:
        # '''
        # Gaussian Blur
        # '''
        if "Gaußscher Unschärfefilter" in methods:
            st.write("""
            ### Gaußscher Unschärfefilter
            """)
            kernel_size = st.slider('Filtergröße', 3, 33, 3, step=2)
            sigma = st.slider('Standardabweichung', 0.1, 15.0, 0.1)
            augmentation.append(T.GaussianBlur(kernel_size=kernel_size, sigma=sigma))

    with col3:
        # '''
        # Color Jitter
        # '''
        if "Farbjitter" in methods:
            st.write("""
            ### Farbjitter
            """)
            brightness = st.slider('Helligkeit', 0.0, 1.0, 1.0)
            contrast = st.slider('Kontrast', 0.0, 1.0, 1.0)
            saturation = st.slider('Sättigung', 0.0, 1.0, 1.0)
            hue = st.slider('Farbton', 0.0, 0.5, 0.0)
            augmentation.append(T.ColorJitter(brightness=[brightness, brightness], contrast=[contrast, contrast], saturation=[saturation, saturation], hue=[hue, hue]))

    with col4:
        # '''
        # Random Cropping
        # '''
        if "Zufälliges Zuschneiden" in methods:
            st.write("""
            ### Zufälliges Zuschneiden
            """)
            size = st.slider('Bildgröße der Ausgabe', 128, 512, 512)
            padding = None
            padding_mode = None
            augmentation.append(T.RandomCrop(size=(size, size), padding=padding, padding_mode=padding_mode))

    with col5:
        if "Spiegeln" in methods:
            st.write("""
            ##### Mirroring
            """)
            st.write('Richtung der Achsenspiegelung')
            if st.checkbox('Horiontal', False):
                aug_img = T.functional.hflip(aug_img)
                aug_img2 = T.functional.hflip(aug_img2)
                aug_img3 = T.functional.hflip(aug_img3)
            if st.checkbox('Vertikal', False):
                aug_img = T.functional.vflip(aug_img)
                aug_img2 = T.functional.vflip(aug_img2)
                aug_img3 = T.functional.vflip(aug_img3)

    with col6:
        if "Rauschen" in methods:
            st.write("""
            ### Rauschen
            """)
            noise_mode = st.radio("Rauschmethode: ", ('none', 'gaussian', 'speckle', 'poisson', 's&p'))  # 'salt', 'pepper', 's&p', 'poisson', 'localvar'
            # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            if noise_mode == 'none':
                pass
            else:
                aug_img = random_noise(np.array(aug_img), mode=noise_mode, clip=True)
                aug_img2 = random_noise(np.array(aug_img2), mode=noise_mode, clip=True)
                aug_img3 = random_noise(np.array(aug_img3), mode=noise_mode, clip=True)

                aug_img = Image.fromarray(((aug_img / aug_img.max())*255).astype(np.uint8))
                aug_img2 = Image.fromarray(((aug_img2 / aug_img2.max())*255).astype(np.uint8))
                aug_img3 = Image.fromarray(((aug_img3 / aug_img3.max())*255).astype(np.uint8))

    st.markdown("---")

    if methods:
        augmentation = torch.nn.Sequential(*augmentation)
        aug_img = augmentation(aug_img)
        aug_img2 = augmentation(aug_img2)
        aug_img3 = augmentation(aug_img3)
        cat_aug_imgs = concat(aug_img, aug_img2, aug_img3)

        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 3, 1])
        with col1:
            st.write("")
        with col2:
            fig, ax = plt.subplots(2)
            ax[0].imshow(cat_orig_imgs, cmap='gray')
            ax[0].axis('off')
            ax[1].axis('off')
            st.pyplot(fig)
        with col3:
            st.image('StreamlitApps/Lektion-3/l2t_images/Pfeil.png', caption='Augmentierung')
        with col4:
            fig3, ax3 = plt.subplots(2)
            ax3[0].imshow(cat_aug_imgs, cmap='gray')
            ax3[0].axis('off')
            ax3[1].axis('off')
            st.pyplot(fig3)
        with col5:
            st.write("")


def metrics(session_state):
    st.write("""
    ## 5. Evaluationsmetriken
    """)
    st.write("""
    ### Segmentierung""")
    st.write("""Die Genauigkeit einer Segmentierung lässt sich über verschiedene Metriken messen. Ein bekanntes Maß für die Qualität einer Segmentierung ist der **Dice-Koeffizient**. Dieser misst die Ähnlichkeit von vorhergesagter Segmentierung und Grundwahrheit, indem ermittelt wird, wie sehr sich die beiden Flächen überlappen. """)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.write("")
    with col2:
        st.image('StreamlitApps/Lektion-3/l2t_images/DiceKoeffizient.png')  # , caption='Vorverarbeitung')
    with col3:
        st.write("")

    st.markdown("""---""")

    if st.checkbox("""Real-Time-Example""", False):
        x = 256
        y = 256
        r = 70

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            #transl_x = st.slider("Translation in X-Richtung", 0, int(r*2)+5, int(r))
            transl_x = st.slider("Translation in X-Richtung", 0, int(r*2)+5, int(r)//2)
            r2 = st.slider("Verhältnis der Kreisradien in %", 30, 100, 100)
            r2 = r * (r2 / 100)
        with col3:
            st.write("")

        background_img = Image.new(mode='RGBA', size=(x, y), color=(255, 255, 255))
        draw = ImageDraw.Draw(background_img)

        draw.ellipse((x/2-r, y/2-r, x/2+r, y/2+r), fill=(114, 178, 216, 180), outline=(0, 162, 255, 255), width=3)

        foreground_img = Image.new(mode='RGBA', size=(x, y), color=(255, 255, 255))
        draw = ImageDraw.Draw(foreground_img)

        draw.ellipse((x/2-r2, y/2-r2, x/2+r2, y/2+r2), fill=(214, 165, 159, 180), outline=(250, 129, 113, 255), width=3)

        foreground_img = T.functional.affine(img=foreground_img, angle=0, translate=(transl_x, 0), scale=1, shear=0, fillcolor=(255, 255, 255, 255))
        overlay = Image.blend(background_img, foreground_img, 0.5)

        with col2:
            st.image(overlay)

        # convert RGBA images to binary segmentation mask
        transform = T.ToTensor()
        background_img = transform(background_img)
        foreground_img = transform(foreground_img)
        background_img = background_img[1, :, :]
        foreground_img = foreground_img[0, :, :]
        background_img[background_img != 1] = 0
        foreground_img[foreground_img != 1] = 0

        iflat = (background_img == 0).view(-1).float()
        tflat = (foreground_img == 0).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        d0 = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
        
        with col1:
            st.write(f'#### Dice: {round(d0.item(),2)}')

    st.markdown("---")
    st.write("""
    ### Klassifikation""")
    st.write("""Accuracy, Sensitivität und Spezifizität wird berechnet um eine Klassifikationsgenauigkeit zu bestimmen. Um diese Maße angeben zu können, wird bestimmt, wie viele Fälle """)
    st.markdown("- korrekterweise als positiv erkannt (= eine bestimmte Klasse vorhanden) wurden (*True Positive*, **TP** )")
    st.markdown("- korrekterweise als negativ erkannt (= eine bestimmte Klasse *nicht* vorhanden) wurden (*True Negative*, **TN**)")
    st.markdown("- fälschlicherweise als positiv erkannt wurden (*False Positive*, **FP**)")
    st.markdown("- fälschlicherweise als negativ erkannt wurden (*False Negative*, **FN**).")
    st.write("""Die **Accuracy** (Treffergenauigkeit) gibt an, wie viele Fälle korrekt klassifiziert wurden. Die **Sensitivität** gibt die Rate der richtig-positiven Klassifikationen an und die **Spezifität** die der richtig-negativen Klassifikationen.""")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.write("")
    with col2:
        st.image('StreamlitApps/Lektion-3/l2t_images/AccSensSpez.png')  # , caption='Vorverarbeitung')
    with col3:
        st.write("")

    st.markdown("""---""")
    
    if st.checkbox("""Eigenes Klassifikationsbeispiel""", False):
        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            st.write('**Einstellen der Anzahl an TP, TN, FP, FN**')

            tp_ex = st.slider("Anzahl TP", 0, 100, 0)
            tn_ex = st.slider("Anzahl TN", 0, 100, 0)
            fp_ex = st.slider("Anzahl FP", 0, 100, 0)
            fn_ex = st.slider("Anzahl FN", 0, 100, 0)
            
            n_samples_ex = tp_ex + tn_ex + fp_ex + fn_ex
            
            st.write('insgesamt ', n_samples_ex, 'Klassifikationen')


        with col3:
            eps = 1e-30
            acc_ex = (tp_ex + tn_ex) / (n_samples_ex + eps)
            sens_ex = tp_ex / (tp_ex+fn_ex + eps)
            spec_ex = tn_ex / (tn_ex+fp_ex + eps)

            st.write('**Ergebnisse: **')
            st.write('*Accuracy* = ', round(acc_ex, 4))
            st.write('*Sensitivität* = ', round(sens_ex, 4))
            st.write('*Spezifität* = ', round(spec_ex, 4))
            

    if st.checkbox("""Zufälliges Klassifikationsbeispiel""", False):
        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            st.write('**Einstellen der Anzahl an Testfällen: **')

            ans = ['benigne', 'maligne']
            KI = []
            KI2 = []
            GT = []
            GT2 = []
            row_names = []
            n_samples = st.slider("""Anzahl an Testbildern""", 5, 50, 10)
            for i in range(n_samples):
                x = random.choice(ans)
                KI.append(x)
                KI2.append(ans.index(x))

                y = random.choice(ans)
                GT.append(y)
                GT2.append(ans.index(y))

            df = pd.DataFrame({"Vorhersage KI":KI, "Grundwahrheit":GT})

            for n in range(n_samples):
                row_names.append(f'Bild Nr. {n+1}')

            df.index = row_names
            st.dataframe(df)

        with col3:
            tn, fp, fn, tp = confusion_matrix(GT2, KI2).ravel()
            acc = (tp + tn) / n_samples
            sens = tp / (tp+fn)
            spec = tn / (tn+fp)

            st.write('**Ergebnisse: **')
            st.write('TP: ', tp, 'TN: ', tn, 'FP: ', fp, 'FN: ', fn)
            st.write('*Accuracy* = ', round(acc, 4))
            st.write('*Sensitivität* = ', round(sens, 4))
            st.write('*Spezifität* = ', round(spec, 4))
            
            
            
    





##############
session_state = st.session_state
if "button_id" not in session_state:
    session_state["button_id"] = ""
if "slider_value" not in session_state:
    session_state["slider_value"] = 0

st.set_page_config(
    page_title="KI Campus: Learn2Trust - Lektion 3", page_icon=":pencil2:"
)

st.title("KI Campus: Learn2Trust ")
st.write("## **Lektion 3: Medizinische Bilddatensätze**")

col1 = st.sidebar
col2, col3 = st.columns((1,1))

col1.subheader("Unterkapitel")
PAGES = {
    "1. Einführung": intro,
    "2. Annotationen": annotations,
    "3. Vorverarbeitung": preprocessing,
    "4. Augmentierung": augmentation,
    "5. Evaluationsmetriken": metrics,

}
page = col1.selectbox("Auswahl:", options=list(PAGES.keys()))


st.sidebar.markdown("""---""")
st.sidebar.write("Dieses Projekt wird bereitgestellt auf der ")
link = '[KI-Campus Website](https://ki-campus.org/)'
st.sidebar.markdown(link, unsafe_allow_html=True)
st.sidebar.image("StreamlitApps/Lektion-3/l2t_images/KICampusLogo.png", use_column_width=True)

PAGES[page](session_state)
