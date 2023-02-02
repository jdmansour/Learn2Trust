import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from segmentation_network import lraspp_mobilenet_v3_large

from torch import nn, Tensor
from typing import Any, Optional

from collections import OrderedDict
from typing import Optional, Dict

from PIL import Image
import matplotlib.animation as animation
import streamlit.components.v1 as components

def cat_images(input,clip=False,size=256):
    if(clip):
        input1 = torch.pow(torch.clamp(input*1000+1500,0,2200)/2200,0.6)
    else:
        input1 = input*1+0
    imcat = torch.cat((input1[:4].view(-1,size),input1[4:8].view(-1,size)),1)
    imcat = torch.cat((imcat[:size*2,:],imcat[size*2:,:]),1)
    return imcat

def color_rgb(image,segment):
    cmap = matplotlib.cm.get_cmap('Set1')
    colors = torch.cat((torch.zeros(1,3),torch.from_numpy(cmap(np.linspace(0,1,9))[:,:3]).float()),0)
    #'red','blue','green','purple','orange','yellow','brown','pink','gray'
    colors = colors[torch.tensor([0,2,7,8,1,3]).long(),:]
    #colors = colors[torch.tensor([0,6,7,1,2,8,3,4]).long(),:]
    seg_rgb = colors[segment]

    img_rgb = image.unsqueeze(2).repeat(1,1,3)
    seg_rgb.view(-1,3)[segment.view(-1)==0,:] = img_rgb.view(-1,3)[segment.view(-1)==0,:]
    return seg_rgb


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_and_seg(test_data=False):
    # load dataset
    if test_data:
        images = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['img'].unsqueeze(1).float()
        segmentations = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['seg'].unsqueeze(1).long()
    else:
        images = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['img'].unsqueeze(1)
        segmentations = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['seg'].long()

    # rescale image values
    images -= 1500
    images /= 1000
    return images, segmentations



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    return torch.load('StreamlitApps/Lektion-5/l2t_data/Learn2Trust_JSRT_LRASPP_finetuned_dict.pth')

def load_fine_model(n_channels_in=1, n_classes=6):
    finetune_model = lraspp_mobilenet_v3_large()
    finetune_model.backbone['0'][0] = nn.Conv2d(n_channels_in, 16, 3, stride=2, padding=1)
    finetune_model.classifier.low_classifier = nn.Conv2d(40, n_classes, 1)
    finetune_model.classifier.high_classifier = nn.Conv2d(128, n_classes, 1)
    finetune_model.load_state_dict(load_model())
    return finetune_model

# @st.cache(suppress_st_warning=True)
def aug_img_and_seg(imgs, imgs_orig, segs_gt, strength_aug_finetune=0.04):
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0) + strength_aug_finetune * torch.randn(imgs.shape[0], 2, 3),
                         (imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]))
    imgs_aug = F.grid_sample(imgs_orig, grid)
    segs_gt_aug = F.grid_sample(segs_gt.float(), grid, mode='nearest').long()
    return imgs_aug, segs_gt_aug


def started():

    st.write("""
    ## 1. Einführung
    """)


    col1, col2 = st.columns([5, 1])

    with col1:
        st.write("""In dieser Lektion geht es darum, wie Künstliche Intelligenz in der medizinischen Bildanalyse dazu eingesetzt werden kann, um medizinische Bildobjekte zu segmentieren.
                """)
        st.write(
            """Am Beispiel von Röntgenthoraxdaten wird demonstriert, wie ein einfaches Segmentierungsnetzwerk programmiert werden kann. Dieses Netzwerk soll entscheiden, wo sich die Lungenflügel, Schlüsselbeine und das Herz befinden. """)

        st.write("""
    **Segmentierung** in der medizinischen Bildverarbeitung bedeutet, dass ein Bild in zusammenhängende Bereiche eingeteilt wird, z.B. zur Abgrenzung diagnostisch oder therapeutisch relevanter Bildbereiche. Hierfür wäre das Beispiel, wenn eine Röntgenthoraxaufnahme in verschiedene Bildbereiche unterteilt wird. Werden die ermittelten Bildbereiche gleichzeitig klassifiziert, so handelt es sich um **semantische Segmentierung**.
Im Falle der Röntgenthoraxaufnahme in unserem Beispiel wäre das Lungenflügel, rechtes und linkes Schlüsselbein und Herz.

    """)


        st.write("""In den verschiedenen Unterkapiteln wird zunächst der Beispieldatensatz gezeigt und anschließend durch Netzwerkerstellung, -training und -evaluation geführt.
                """)

    with col2:
        st.image('StreamlitApps/Lektion-5/l2t_images/intro5.png')



    st.markdown("---")

    col1, col2, col3 = st.columns([1,1,1.5])
    with col1:
        st.write("")
    with col2:
        st.write("""
        ### Unterkapitel
        Auswahl über Seitenleiste

        1. Einführung
        2. Datensatz
        3. Netzwerkarchitektur
        4. Finetuning
        5. Evaluation
        6. Ergebnisvisualisierung
        7. Gelernte Merkmale
        """)
    with col3:
        st.write("")

    st.markdown("---")




def dataset():

    st.write("""
    ## 2. Datensatz
    """)

    st.markdown("""
        Der Datensatz für diese Lektion ist aus der JSRT Datenbank (Japanese Society of Radiological Technology):

        *Shiraishi, Junji, et al. "Development of a digital image database for chest radiographs with and without a lung nodule: receiver operating
                  characteristic analysis of radiologists' detection of pulmonary nodules." American Journal of Roentgenology 174.1 (2000): 71-74.*

                  """)

    st.markdown("""
        Dieser Datensatz enthält insgesamt 247 konventionelle Röntgenaufnahmen des Thorax.
                """)

    col0, col1, col2 = st.columns([10,30,15])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/img_seg.png', width=400)

    col0, col1, col2 = st.columns([15,2,30])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/blau.png', width=25)
    with col2:
        st.write('  rechter Lungenflügel')

    col0, col1, col2 = st.columns([15,2,30])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/braun.png', width=25)
    with col2:
        st.write('  linker Lungenflügel')

    col0, col1, col2 = st.columns([15,2,30])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/rosa.png', width=25)
    with col2:
        st.write('  rechtes Schlüsseslbein')

    col0, col1, col2 = st.columns([15,2,30])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/rot.png', width=25)
    with col2:
        st.write('  linkes Schlüsselbein')

    col0, col1, col2 = st.columns([15,2,30])
    with col1:
        st.image('StreamlitApps/Lektion-5/l2t_images/gruen.png', width=25)
    with col2:
        st.write('  Herz')



    st.markdown("""---""")


    st.write("""
    #### Visualisierung und Augmentierung des Datensatzes
    """)
    st.markdown("""Wenn der Datensatz geladen wurde, werden Beispielbilder visualisiert und optional Segmentierungen angezeigt, die von Experten erstellt wurden und als Grundwahrheit
        genutzt werden.
        """)

    if st.checkbox("Datensatz laden", False):

        st.write('**Beispiele aus dem Bilddatensatz**')
        # load data
        images, segmentations = load_data_and_seg()

        value = 0  # default
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.checkbox("Label anzeigen", False):
                value = st.slider('Transparenz für Label einstellen', 0, 100, 40)
        with col2:
            if st.checkbox("Datensatz augmentieren", False):
                strength = st.slider('Stärke der Augmentierung einstellen',0.00, 0.10, 0.04)

                images = images.detach().clone()
                segmentations = segmentations.detach().clone()

                # define random affine grid for data augmentation (strength defined by slider value above)
                grid = F.affine_grid(torch.eye(2,3).unsqueeze(0)+strength*torch.randn(8,2,3),(8,1,images.shape[2],images.shape[3]))

                # warp images and segmentations to be visualized with generated grid
                images[:8] = F.grid_sample(images[:8],grid)
                segmentations[:8] = F.grid_sample(segmentations[:8].unsqueeze(1).float(),grid, mode='nearest').squeeze(1).long()

        #visalise images with segmentations
        imcat = cat_images(images[:8,0],clip=True)
        segcat = cat_images(segmentations[:8])
        seg_rgb = color_rgb(imcat,segcat)

        fig, ax = plt.subplots()
        ax.imshow(imcat,'gray')
        ax.imshow(seg_rgb,alpha=float(value)*0.01, interpolation='None')
        ax.axis('off')
        st.pyplot(fig)


def architecture():
    st.write("""
    ## 3. Netzwerkarchitektur
    """)

    st.markdown("""
    Die Netzwerkarchitektur ist das sogenannte *MobileNetV3*, vorgestellt in

       *Howard, Andrew, et al. "Searching for mobilenetv3." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.*

    Link zu Preprint: https://arxiv.org/abs/1905.02244
    """)

    st.write("""Dieses Faltungsnetzwerk zeichnet sich dadurch aus, dass es sehr effizient ist und gleichzeitig vergleichsweise wenige Parameter besitzt, was bedeutet, dass es schnell zu trainieren ist und wenig Speicherplatz beansprucht.
Es besteht aus einigen Schichten, welche Merkmale für die Segmentierung extrahieren und am Ende eine Schicht, welche final die Klassenzuordnung der segmentierten Bildbereiche entscheidet.
    """)

    col1, col2, col3 = st.columns([1, 10, 1])

    with col2:
        st.image('StreamlitApps/Lektion-5/l2t_images/MobileNetPretrain.png')


    st.write("""
        In dieser Lektion wird eine bereits vortrainierte Version des Deep-Learning-Modell verwendet. Dieses hat bereits gelernt, den rechten Lungenflügel, das rechte Schlüsselbein und das Herz zu segmentieren. """)

    if st.checkbox("""Vorhersagen des vortrainierten Modells anzeigen""", False):
        images, segmentations_6 = load_data_and_seg()
        segmentations = segmentations_6.clone()
        segmentations[segmentations_6==4] = 0
        segmentations[segmentations_6==2] = 0

        value = 0  # default
        value = st.slider('Transparenz für Label einstellen', 0, 100, 40)

        #visalise images with segmentations
        imcat = cat_images(images[:8,0],clip=True)
        segcat = cat_images(segmentations[:8])
        seg_rgb = color_rgb(imcat,segcat)

        fig, ax = plt.subplots()
        ax.imshow(imcat,'gray')
        ax.imshow(seg_rgb,alpha=float(value)*0.01, interpolation='None')
        ax.axis('off')
        st.pyplot(fig)

    st.markdown("""---""")

    # st.write("""
    # ### Laden des Modells
    # """)

    st.write("""
        Das vortrainierte Modell zu nutzen und anzupassen, um zusätzlich auch den linken Lungenflügel und das linke Schlüsselbein zu segmentieren, führt zu einem deutlich schnelleren Lernprozess als wenn das Modell komplett neu trainiert werden müsste.
        """)

    if st.checkbox("""Modell modifizieren für zusätzliche Labelklassen""", False):


        st.write("""Es wird ein vortrainiertes Netzwerkmodell geladen und für die Segmentierung von zwei weiteren Klassen modifiziert.""")

        col1, col2, col3 = st.columns([1, 10, 1])

        with col2:
            st.image('StreamlitApps/Lektion-5/l2t_images/MobileNet.png')



        st.write('**Code:**')

        with st.echo():

            #load model
            model = lraspp_mobilenet_v3_large()

            # adapt model: 6 label classes and 1 input channel
            n_channels_in = 1 # number of input channels
            n_classes = 6 # number of label classes

            # adapt first layer -> correct number of input channels
            model.backbone['0'][0] = nn.Conv2d(n_channels_in, 16, 3, stride=2, padding=1)

            # adapt classification layers -> correct number of label classes
            model.classifier.low_classifier = nn.Conv2d(40, n_classes, 1)
            model.classifier.high_classifier = nn.Conv2d(128, n_classes, 1)



def finetune():
    st.write("""## 4. Finetuning""")

    imgs, _ = load_data_and_seg()

    st.write("""Da das geladene Netzwerk bereits vortrainiert wurde, muss es im Folgenden durch ein erneutes Training nur noch
    verfeinert beziehungsweise auf die Segmentierung von zwei weiteren Klassen angepasst werden. Dieser Prozess wird als **Finetuning** bezeichnet. Der Vorteil, den das Finetuning bietet ist, dass das Modell bereits viele Muster gelernt hat und das Training mit weniger Daten auskommt.""")

    if st.checkbox("""Finetuning""", False):
        finetune_model = load_fine_model()
        finetune_model.train()

        # load test data
        imgs_testset = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['img'].unsqueeze(1).float()
        segs_testset = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['seg'].long()
        imgs_testset = imgs_testset.clone()

        # rescale image values
        imgs_testset = (imgs_testset - 1500) / 1000

        st.write('**Definition verschiedener Trainingsparameter für das Finetuning:**')
        st.write('Zum Finetuning des Modells wird über 2500 Epochen mit dem Adam-Optimierer und einer Lernrate von 0,001 trainiert.')
        st.write('**Code:**')

        st.code('''
        # number of epochs for fine-tuning
        n_epochs = 2500

        # initialize optimizer
        optimizer = torch.optim.Adam(model_finetuning.parameters(),lr=0.001)
        ''')

        st.markdown("---")

        st.write('**Trainingsschleife für das Finetuning:**')
        st.write('Für das Finetuning wird eine Batch-Größe von acht gewählt. Das bedeutet, dass in jeder Epoche acht zufällige Bilder aus dem Finetuning-Datensatz ausgewählt werden. Diese werden augmentiert und dem Modell präsentiert (*forward pass*). Mithilfe eines Kreuzentropie-Losses wird die Abweichung zwischen Vorhersage des Modells und der Grundwahrheit ermittelt. Anschließend werden die Gewichte des Modells im *backward pass* modifiziert.')
        st.write('**Code:**')

        st.code('''
    model_finetuning.train()

    for epoch in range(n_epochs):
        idx = torch.randperm(10)[:8]
        optimizer.zero_grad()

        # load data for fine-tuning
        img = imgs_finetune[idx]
        seg = segs_finetune[idx]

        # perform data augmentation
        img_aug, seg_aug  = aug_img_and_seg(img, seg)

        # forward pass
        predict = model_finetuning(img_aug)

        # compute loss
        loss = nn.CrossEntropyLoss()(predict['out'], seg_aug)

        # backward pass
        loss.backward()
        optimizer.step()
        ''')


        st.write("Die Loss-Kurve, die während des Finetunings durch den obigen Trainings-Loop entsteht, ist hier abgebildet:")
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.image('StreamlitApps/Lektion-5/l2t_images/loss_curve.png')


        st.markdown("---")

        if st.checkbox("""Anwendung auf Testbilder""", False):

            st.write("** *Forward pass* für acht Testbilder**:")
            st.write("Nach dem Finetuning ist das Modell dazu in der Lage, auf ungesehenen Testbildern Vorhersagen zu treffen. Im nächsten Unterkapitel werden Inferenz und Evaluation näher erläutert. ")
            st.write('**Code:**')

            with st.echo():
                # forward pass for eight test samples
                with torch.no_grad():

                    # pass through model backbone
                    features = finetune_model.backbone(imgs_testset)

                    # pass through classifier and argmax()
                    prediction = F.interpolate(finetune_model.classifier(features), scale_factor=8, mode='bilinear').argmax(1)

            st.write("**Visualisierung der Vorhersagen**:")

            imcat = cat_images(imgs_testset[:8, 0], clip=True)
            segcat = cat_images(prediction[:8])
            seg_rgb = color_rgb(imcat, segcat)

            fig, ax = plt.subplots()
            ax.imshow(imcat, 'gray')
            ax.imshow(seg_rgb, alpha=float(40) * 0.01, interpolation='None')
            ax.axis('off')
            st.pyplot(fig)


def evaluation():
    st.write("""
    ## 5. Evaluation
    """)

    st.markdown(
        "Im vorherigen Unterkapitel *Training* wurde das Modell einem erneuten Training unterzogen. Dieses Modell soll jetzt während der sogenannten **Inferenz** auf die Testdaten angewendet werden, um zu evaluieren, wie gut das Modell die Bilder segmentieren kann, die ihm während der Trainingsphase noch nicht präsentiert wurden.")
    st.write("Während der Evaluation wird der Dice-Koeffizient bestimmt, der die Genauigkeit der Segmentierung angibt.")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.write("")
    with col2:
        st.image('StreamlitApps/Lektion-5/l2t_images/DiceKoeffizient.png')  # , caption='Vorverarbeitung')
    with col3:
        st.write("")

    st.markdown("---")
    st.markdown("""Bei Starten der Inferenz werden""")
    st.markdown("(1) die Testbilder und deren Grundwahrheiten geladen")
    st.markdown("(2) das gespeicherte Modell auf die Testbilder angewendet")
    st.markdown("(3) die Vorhersagen des Modells mit den Grundwahrheiten verglichen")

    if st.checkbox("""Inferenz starten""", False):
        imgs, segs_gt = load_data_and_seg(test_data=True)
        imgs_orig = imgs.clone()

        model = load_fine_model()#

        st.markdown('Je nach Augemtierungsstärke variieren die Evaluations-Ergebnisse für die Testbilder.')
        strength = st.slider('Stärke der Augmentierung einstellen',0.00, 0.10, 0.02)

        imgs_aug, segs_gt_aug = aug_img_and_seg(imgs, imgs_orig, segs_gt, strength)

        model.train()
        with torch.no_grad():
            features = model.backbone(imgs_aug)
            prediction = F.interpolate(model.classifier(features), scale_factor=8, mode='bilinear').argmax(1)

        d0 = torch.zeros(10, 5)
        for i in range(10):
            d0[i] = dice_coeff(segs_gt_aug[i].squeeze(0).contiguous(), prediction[i].contiguous(), 6)
            # st.write(i, d0[i])
        st.markdown('Mittelwert Dice-Koeffizient über alle Labelklassen:  ```{:.2f}```'.format(d0.mean().item()))
        st.markdown('Mittelwert Dice-Koeffizient für einzelne Labelklassen: ')
        col0, col1, col2 = st.columns([2, 1, 30])
        with col0:
            st.write("")
        with col1:
            st.image('StreamlitApps/Lektion-5/l2t_images/blau.png', width=15)
        with col2:
            st.markdown('*Label 1*: rechter Lungenflügel:  ```{:.2f}```'.format(np.asarray(d0.mean(0))[0]))
        col0, col1, col2 = st.columns([2, 1, 30])
        with col0:
            st.write("")
        with col1:
            st.image('StreamlitApps/Lektion-5/l2t_images/braun.png', width=15)
        with col2:
            st.markdown('*Label 2*: linker Lungenflügel:  ```{:.2f}```'.format(np.asarray(d0.mean(0))[1]))
        col0, col1, col2 = st.columns([2, 1, 30])
        with col0:
            st.write("")
        with col1:
            st.image('StreamlitApps/Lektion-5/l2t_images/rosa.png', width=15)
        with col2:
            st.markdown('*Label 3*: rechtes Schlüsselbein:  ```{:.2f}```'.format(np.asarray(d0.mean(0))[2]))
        col0, col1, col2 = st.columns([2, 1, 30])
        with col0:
            st.write("")
        with col1:
            st.image('StreamlitApps/Lektion-5/l2t_images/rot.png', width=15)
        with col2:
            st.markdown('*Label 4*: linkes Schlüsselbein:  ```{:.2f}```'.format(np.asarray(d0.mean(0))[3]))
        col0, col1, col2 = st.columns([2, 1, 30])
        with col0:
            st.write("")
        with col1:
            st.image('StreamlitApps/Lektion-5/l2t_images/gruen.png', width=15)
        with col2:
            st.markdown('*Label 5*: Herz:  ```{:.2f}```'.format(np.asarray(d0.mean(0))[4]))


def evaluation_visualisation():
    st.write("""
            ## 6. Ergebnisvisualisierungen
            """)

    st.write("""
    Hier wird die Netzwerkausgabe für die (unaugmentierten) Testbilder der Grundwahrheit gegenüber gestellt.""")

    st.write(f"""
    Insgesamt wurden {10} Bilder segmentiert.""")

    imgs, segs_gt = load_data_and_seg(test_data=True)
    imgs_orig = imgs.clone()
    model = load_fine_model()
    imgs_aug, segs_gt_aug = aug_img_and_seg(imgs, imgs_orig, segs_gt, 0.0)

    model.train()
    with torch.no_grad():
        features = model.backbone(imgs_aug)
        prediction = F.interpolate(model.classifier(features), scale_factor=8, mode='bilinear').argmax(1)

    col1, col2 = st.columns([1,1])
    with col1:
        idx = st.slider("Bildindex", 0, len(imgs_aug)-1, 0)
    with col2:
        label_transparency = st.slider("Transparenz für Label einstellen",0, 100, 40)

    img_aug = torch.pow(torch.clamp(imgs_aug[idx, 0] * 1000 + 1500, 0, 2200) / 2200, 0.6)
    seg_gt = color_rgb(img_aug, segs_gt_aug[idx])
    seg_pred = color_rgb(img_aug, prediction[idx])
    d0 = dice_coeff(segs_gt_aug[idx].squeeze(0).contiguous(), prediction[idx].contiguous(), 6)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(img_aug, 'gray')
        ax.imshow(X=seg_gt.squeeze(), alpha=float(label_transparency)*0.01, interpolation='None')
        ax.axis('off')
        ax.set_title('Grundwahrheit')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(img_aug, 'gray')
        ax.imshow(X=seg_pred.squeeze(), alpha=float(label_transparency)*0.01, interpolation='None')
        ax.axis('off')
        ax.set_title('Netzwerkausgabe')
        st.pyplot(fig)

    st.markdown("Dice-Koeffizient: ```{:.2f}```".format(d0.mean().item()))

def features():
    st.write("""
    ## 7. Gelernte Merkmale
    """)

    model = lraspp_mobilenet_v3_large()
    n_channels_in = 1  # number of input channels
    n_classes = 6  # number of label classes

    # adapt first layer -> correct number of input channels
    model.backbone['0'][0] = nn.Conv2d(n_channels_in, 16, 3, stride=2, padding=1)

    # adapt classification layers -> correct number of label classes
    model.classifier.low_classifier = nn.Conv2d(40, n_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(128, n_classes, 1)

    st.write("Das verwendete Deep-Learning-Modell nutzt für die Klassenzuordnung in der letzten Schicht sowohl Merkmale in niedriger Auflösung, aber mit vielen Merkmalskanälen (*features_high*) als auch Merkmale in hoher Auflösung, aber mit wenigen Merkmalskanälen (*features_low*).")

    st.image('StreamlitApps/Lektion-5/l2t_images/gelernteMerkmale.png')



    if st.checkbox("Laden von trainiertem Finetune-Modell und Testbildern", False):
        # load trained weights
        state_dict = torch.load('StreamlitApps/Lektion-5/l2t_data/Learn2Trust_JSRT_LRASPP_finetuned_dict.pth')
        model.load_state_dict(state_dict)

        # load test image data
        images = torch.load('StreamlitApps/Lektion-5/l2t_data/jsrt_img_and_seg_test.pth')['img'].unsqueeze(1)

        # rescale test images
        images -= 1500
        images /= 1000

        with torch.no_grad():
            # pass through model backbone
            output = model.backbone(images[:8])

            features_low = output['low']
            features_high = output['high']


        st.write("#### **Visualisierung der gelernten Merkmale**")
        st.write("Über Slider jeweiligen Merkmalskanal auswählen für Visualisierung")

        channel_low = st.slider('Angezeigter Merkmalskanal (features_low)', 0, features_low.shape[1] - 1, 0)

        imcat = cat_images(features_low[:8, channel_low], clip=False, size=32)

        fig, ax = plt.subplots()
        ax.imshow(imcat)
        ax.axis('off')
        st.pyplot(fig)

        channel_high = st.slider('Angezeigter Merkmalskanal (features_high)', 0, features_high.shape[1] - 1, 0)

        imcat = cat_images(features_high[:8, channel_high:(channel_high + 1)].mean(1), clip=False, size=16)

        fig, ax = plt.subplots()
        ax.imshow(imcat)
        ax.axis('off')
        st.pyplot(fig)


st.set_page_config(
    page_title="KI Campus: Learn2Trust - Lektion 5", page_icon=":pencil2:"
)

st.title("KI Campus: Learn2Trust ")
st.write("## **Lektion 5: Semantische Segmentierung**")

st.sidebar.subheader("Unterkapitel")
PAGES = {
    "1. Einführung": started,
    "2. Datensatz": dataset,
    "3. Netzwerkarchitektur": architecture,
    "4. Finetuning": finetune,
    "5. Evaluation" : evaluation,
    "6. Ergebnisvisualisierung": evaluation_visualisation,
    "7. Gelernte Merkmale": features
}
page = st.sidebar.selectbox("Auswahl:", options=list(PAGES.keys()))

st.sidebar.markdown("""---""")
st.sidebar.write("Dieses Projekt wird bereitgestellt auf der ")
link = '[KI-Campus Website](https://ki-campus.org/)'
st.sidebar.markdown(link, unsafe_allow_html=True)
st.sidebar.image("StreamlitApps/Lektion-5/l2t_images/KICampusLogo.png", use_column_width=True)
PAGES[page]()
