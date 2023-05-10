# Imports
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from classificationNet import classificationCNN
from sklearn.metrics import confusion_matrix

from annotated_text import annotated_text

import copy


# Definition of functions and variables
def cat_images(input, clip=False, size=128):
    if clip:
        input1 = torch.pow(torch.clamp(input * 1000 + 1500, 0, 2200) / 2200, 0.6)
    else:
        input1 = input * 1 + 0
    imcat = torch.cat((input1[:4].view(-1, size), input1[4:8].view(-1, size)), 1)
    imcat = torch.cat((imcat[: size * 2, :], imcat[size * 2 :, :]), 1)
    return imcat


def label2text(label):
    if label == 0:
        label_text = "keine Pneumonie"
    if label == 1:
        label_text = "Pneumonie"
    return label_text


@st.cache(suppress_st_warning=True)
def load_data():
    return torch.from_numpy(np.load('StreamlitApps/Lektion-4/l2t_data/pneumonia_detection_data_img.npz')['arr_0']).float()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_quiz():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/data_highres.pth")


@st.cache(suppress_st_warning=True)
def load_data_label():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/pneumonia_detection_data_label.pth")

@st.cache(suppress_st_warning=True)
def load_data_label_quiz():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/data_label_highres.pth")


@st.cache(suppress_st_warning=True)
def load_idx_train():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/idx_train.pth")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_idx_quiz():
    return np.random.randint(20,size=(20))


@st.cache(suppress_st_warning=True)
def load_idx_val():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/idx_val.pth")


@st.cache(suppress_st_warning=True)
def load_idx_test():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/idx_test.pth")


@st.cache(suppress_st_warning=True)
def load_model():
    return torch.load("StreamlitApps/Lektion-4/l2t_data/net_pneumonia_classification.pth")


@st.cache(
    suppress_st_warning=True,
    hash_funcs={matplotlib.figure.Figure: hash},
    # allow_output_mutation=True,
)
def create_figure_example_data(data_img, data_label, idx_train, show_labels):
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)

    if show_labels:
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(data_img[idx_train[i], 0], "gray")
            ax.set_title(label2text(data_label[idx_train[i]]), fontsize=8)
            ax.axis("off")
    else:
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(data_img[idx_train[i], 0], "gray")
            ax.axis("off")

    plt.subplots_adjust(wspace=0.2, hspace=-0.2)

    return fig


#############################################################
########################### Intro ###########################
#############################################################


def intro(session_state):
    st.write("""## 1. Einführung""")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            """In dieser Lektion geht es darum, wie Künstliche Intelligenz (KI) in der medizinischen Bildanalyse dazu eingesetzt werden kann, um Klassifikationsentscheidungen zu treffen."""
        )
        st.write(
            """Am Beispiel von Röntgenthoraxdaten, die zur Diagnostik von Pneumonie aquiriert wurden, wird demonstriert, wie ein einfaches Klassifikationsnetzwerk programmiert werden kann.
            Dieses Klassifikationsnetzwerk soll entscheiden, ob in einem präsentierten Eingabebild eine Pneumonie zu erkennen ist oder nicht. """
        )
        st.write(
            """In den verschiedenen Unterkapiteln wird zunächst der Beispieldatensatz gezeigt und anschließend durch Netzwerkerstellung, -training und -evaluation geführt."""
        )
    with col2:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/intro.png")

    st.markdown("---")

    col1, col2, col3 = st.columns([1,1,1.5])

    with col2:
        st.write(
            """
            ## Unterkapitel
            Auswahl über Seitenleiste

            1. Einführung
            2. Datensatz
            3. Labelquiz
            4. Netzwerkarchitektur
            5. Training
            6. Evaluation
            7. Ergebnisvisualisierung
            """
        )

#st.markdown("---")

#link = "[Notebook](https://drive.google.com/file/d/1dOWX1qQCEzhAFOPQqH7P-V_aVVmC-0e-/view?usp=sharing)"
#st.write(
#"""Programmier-interessierte Lernende haben die Möglichkeit, sich hier tiefergehend mit dem dieser Lektion zugrunde liegenden Code zu befassen:"""
#)
#st.markdown(link, unsafe_allow_html=True)


###############################################################
########################### Dataset ###########################
###############################################################


def dataset(session_state):
    st.write("""## 2. Datensatz""")

    st.write(
        """Der Datensatz für diese Lektion besteht aus Röntgenaufnahmen des Thorax, die erstellt wurden, um Pneumonie (viral oder bakteriell) zu diagnositizieren und stammt von https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.
        Wenn der Datensatz geladen wurde, werden Beispielbilder visualisiert und optional Label angezeigt, die angeben, ob Experten bei den vorliegenden Aufnahmen eine Pneumonie (egal ob viral oder bakteriell) diagnostiziert haben oder nicht."""
    )

    st.markdown("""---""")

    if st.checkbox("""Datensatz laden""", False):
        data_img = load_data()
        data_label = load_data_label()
        idx_train = load_idx_train()
        idx_val = load_idx_val()
        idx_test = load_idx_test()

        st.write("**Beispiele aus dem Bilddatensatz**")

        if st.checkbox("""Label anzeigen""", False):
            fig = create_figure_example_data(data_img, data_label, idx_train, True)
            st.pyplot(fig)
        else:
            fig = create_figure_example_data(data_img, data_label, idx_train, False)
            st.pyplot(fig)

        st.write(
            "Der Datensatz besteht aus insgesamt 5000 Bildern. Davon sind 3600 Röntgenbilder Patienten zuzuordnen, bei denen eine Pneumonie diagnostiziert wurde und 1400 Röntgenbilder Patienten zuzuordnen, bei denen keine Pneumonie festgestellt wurde. ",
        )
        st.write(
            "Der gesamte Bilddatensatz wird aufgeteilt in ",
            len(idx_train),
            " Trainingsbilder, ",
            len(idx_val),
            " Validierungsbilder und ",
            len(idx_test),
            " Testbilder. ",
        )

    st.markdown("""---""")

    st.write(
        """Wenn alle Daten geladen wurden, wird der Datensatz aufgeteilt in Trainingsbilder, Validierungsbilder und Testbilder."""
    )

    col1, col2, col3 = st.columns([1,12,1])
    with col2:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/datasplit.png")

    st.write(
        """Die **Trainingsbilder** werden dem Klassifikationsnetzwerk während des Trainings wiederholt präsentiert und dienen dazu, dass das Modell dadurch lernt.
        Mithilfe der **Validierungsbilder** wird während des Trainingsvorganges validiert, wie gut das Modell mit Eingabedaten umgehen kann, die es nicht zum Lernen verwendet hat.
        Anhand der **Testbilder** wird nach abgeschlossenem Training evaluiert, wie gut das Modell Daten klassifizieren kann, die während des Trainings weder als Trainings- noch als Validierungsdaten gedient haben."""
    )


#################################################################
########################### Labelquiz ###########################
#################################################################
def labelquiz(session_state):
    st.write("""## 3. Labelquiz""")

    data_img = load_data_quiz()
    data_label = load_data_label_quiz()
    idx_train = load_idx_quiz()

    # quiz params
    max_n_question = 5

    # initialization of session state
    if "n_question" not in st.session_state:
        st.session_state.n_question = 0
    if "ans" not in st.session_state:
        st.session_state.ans = []
    if "phase" not in st.session_state:
        st.session_state.phase = 0
    if "quiz_img_idx" not in st.session_state:
        st.session_state.quiz_img_idx = []

    st.write(
        'Klassifiziere die gezeigten Aufnahmen nach "Pneumonie"/"Keine Pneumonie".'
    )

    if len(st.session_state.quiz_img_idx) == 0:
        idx = torch.randperm(len(idx_train))[:max_n_question]
        st.session_state.quiz_img_idx = idx_train[idx]

    # load img and normalize
    img = np.asarray(
        data_img[st.session_state.quiz_img_idx[st.session_state.n_question].item()]
    )
    img = img - img.min()
    img = img / img.max()

    st.write(f"Versuch Nr: {st.session_state.n_question+1}/{max_n_question}")

    # progress bar
    if st.session_state.n_question != 0:
        progress = (st.session_state.n_question) / max_n_question
    else:
        progress = 0
    progress_bar = st.progress(progress)

    # devide page into two columns
    col1, col2 = st.columns([1, 0.4])
    with col1:
        st.image(
            img,
            width=420,
            caption=f"Bild Nr. {st.session_state.quiz_img_idx[st.session_state.n_question].item()}",
        )
    with col2:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/fragezeichen.png", width=120)
        st.write(" ")
        st.write(" ")

        ###################
        # Startbildschirm #
        ###################
        if st.session_state.phase == 0:
            # start button which enters input phase
            if st.button("Starte eine neue Runde."):
                st.session_state.phase += 1
                st.experimental_rerun()
                #raise RerunException(RerunData(widget_states=None))

        ###########
        # Eingabe #
        ###########
        elif st.session_state.phase == 1:
            # checkboxes for user to select one of two classes
            butt_nop = st.checkbox("keine Pneumonie", value=False)
            butt_pne = st.checkbox("Pneumonie", value=False)

            # if a checkbox is selected the next phase is entered
            if butt_nop or butt_pne:
                # calc and save the result
                label = data_label[
                    st.session_state.quiz_img_idx[st.session_state.n_question]
                ]
                result = (
                    "TP"
                    if (butt_pne and label)
                    else "FP"
                    if (butt_pne and not label)
                    else "TN"
                    if (butt_nop and not label)
                    else "FN"
                    if (butt_nop and label)
                    else "ERROR"
                )
                st.session_state.ans.append(result)
                st.session_state.phase += 1
                st.experimental_rerun()
#                raise RerunException(RerunData(widget_states=None))

        ###########
        # Ausgabe #
        ###########
        elif st.session_state.phase == 2:
            result = st.session_state.ans[-1]
            result_answer = (
                'Du hast "Pneumonie" gewählt, das ist korrekt!'
                if result == "TP"
                else 'Du hast "Pneumonie" gewählt, das ist leider nicht korrekt.'
                if result == "FP"
                else 'Du hast "keine Pneumonie" gewählt, das ist korrekt!'
                if result == "TN"
                else 'Du hast "keine Pneumonie" gewählt, das ist leider nicht korrekt.'
                if result == "FN"
                else "ERROR"
            )
            st.write(result_answer)

            # check if last images or not
            if st.session_state.n_question + 1 < max_n_question:
                # go back to input phase
                if st.button("Nächstes Bild"):
                    st.session_state.phase -= 1
                    st.session_state.n_question += 1
                    #raise RerunException(RerunData(widget_states=None))
            else:
                # enter result display phase
                if st.button("Ergebnis"):
                    st.session_state.phase += 1
                    st.experimental_rerun()
                    raise RerunException(RerunData(widget_states=None))

        ######################
        # Ergebnisauswertung #
        ######################
        elif st.session_state.phase == 3:
            # set progress bar to 100%
            progress_bar.progress(1.0)

            # calc acc
            result = st.session_state.ans
            tp, fp, tn, fn = [
                result.count("TP"),
                result.count("FP"),
                result.count("TN"),
                result.count("FN"),
            ]
            acc = (tp + tn) / (tp + fp + tn + fn)

            # print out
            st.write("Ergebnis:")
            st.write(f"TP: {tp}; FP: {fp}; TN: {tn}; FN: {fn}")
            st.write(f"ACCURACY: {acc}")

            # balloons for positive reinforcement
            st.balloons()

            # restart button with resets all session state params
            if st.button("Neustart"):
                st.session_state.n_question = 0
                st.session_state.ans = []
                st.session_state.phase = 0
                st.session_state.quiz_img_idx = []
                #raise RerunException(RerunData(widget_states=None))


####################################################################
########################### Architecture ###########################
####################################################################


def architecture(session_state):
    st.write("""## 4. Netzwerkarchitektur""")

    n_in = 1
    n_out_tmp = 1
    n_out = 1
    n_classes = 2
    n_ch_in = 1
    n_ch_out = 10
    kernel_size = 3

    st.write(
        "Das Modell, das in dieser Lektion zur Klassifikation von Röntgenthoraxbildern verwendet werden soll, ist ein Faltungsnetzwerk bestehend aus vier Blöcken mit Faltungen, Batch-Normalisierungen und Aktivierungen, gefolgt von einem Modul aus voll-verbundenen Schichten."
    )

    col1, col2, col3 = st.columns([1, 15, 1])
    with col1:
        st.write("")
    with col2:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/architektur.png", caption="Netzwerkarchitektur")
    with col3:
        st.write("")

    st.write(
        "In diesem Unterkapitel werden die einzelnen Bausteine des Klassifikationsnetzwerkes beschrieben und ein Einblick in deren Programmierung gegeben."
    )

    st.markdown("""---""")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/doppelconvblock.png", caption="Faltungsblock")
    with col2:
        st.write("**Faltungsblöcke**")
        st.write(
            """Die insgesamt vier Faltungsblöcke bestehen jeweils aus zweimal der Abfolge einer zweidimensionalen Faltung, gefolgt von einer Batch-Normalisierung und einer Aktivierung.
            Auf jeden der Faltungsblöcke folgt ein Max-Pooling. In jedem Faltungsblock erhöht sich die Anzahl der Merkmalskanäle, während jede Pooling-Operation die räumliche Auflösung reduziert."""
        )

    st.write("*Code für einen Faltungsblock:*")
    with st.echo():
        conv_block = nn.Sequential(
            nn.Conv2d(n_ch_in, n_ch_out, kernel_size),
            nn.BatchNorm2d(n_ch_out),
            nn.ReLU(),
            nn.Conv2d(n_ch_out, n_ch_out, kernel_size),
            nn.BatchNorm2d(n_ch_out),
            nn.ReLU(),
        )

    st.markdown("---")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("**Modul aus voll-verbundenen Schichten**")
        st.write(
            """Das Klassifikationsmodell wird durch ein Modul bestehend aus voll-verbundenen Schichten abgeschlossen.
            Dabei wechseln sich lineare Transformationen mit Aktivierungsfunktionen ab.
            Die Ausgabe der letzten Schicht besitzt zwei Merkmalskanäle - entsprechend der Anzahl der Klassen (*Pneumonie/keine Pneumonie*)."""
        )
    with col2:
        st.image("StreamlitApps/Lektion-4/l2t_imgs/fullyconnected.png", caption="Voll-verbundene Schichten")

    st.write("*Code für das Modul aus voll-verbundenen Schichten:*")
    with st.echo():
        fc_block = nn.Sequential(
            nn.Linear(n_in, n_out_tmp),
            nn.ReLU(),
            nn.Linear(n_out_tmp, n_out),
            nn.ReLU(),
            nn.Linear(n_out_tmp, n_classes),
        )

    st.markdown("""---""")

    st.write("**Für Coding-interessierte Lernende:**")
    if st.checkbox("""Gesamten Code für Klassifikationsmodell anzeigen""", False):
        st.write("**Code:**")

        with st.echo():

            class classificationCNN(nn.Module):
                def __init__(self):
                    super().__init__()

                    self.conv_block0 = nn.Sequential(
                        nn.Conv2d(1, 10, 3, bias=False),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                        nn.Conv2d(10, 10, 3, bias=False),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                    )

                    self.conv_block1 = nn.Sequential(
                        nn.Conv2d(10, 20, 3, bias=False),
                        nn.BatchNorm2d(20),
                        nn.ReLU(),
                        nn.Conv2d(20, 20, 3, bias=False),
                        nn.BatchNorm2d(20),
                        nn.ReLU(),
                    )

                    self.conv_block2 = nn.Sequential(
                        nn.Conv2d(20, 40, 3, bias=False),
                        nn.BatchNorm2d(40),
                        nn.ReLU(),
                        nn.Conv2d(40, 40, 3, bias=False),
                        nn.BatchNorm2d(40),
                        nn.ReLU(),
                    )

                    self.conv_block3 = nn.Sequential(
                        nn.Conv2d(40, 80, 3, bias=False),
                        nn.BatchNorm2d(80),
                        nn.ReLU(),
                        nn.Conv2d(80, 80, 3, bias=False),
                        nn.BatchNorm2d(80),
                        nn.ReLU(),
                    )

                    self.fc_block = nn.Sequential(
                        nn.Linear(4 * 4 * 80, 120),
                        nn.ReLU(),
                        nn.Linear(120, 40),
                        nn.ReLU(),
                        nn.Linear(40, 2),
                    )

                    self.maxPool = nn.MaxPool2d(2)

                def forward(self, x):

                    x = self.maxPool(self.conv_block0(x))
                    x = self.maxPool(self.conv_block1(x))
                    x = self.maxPool(self.conv_block2(x))
                    x = self.maxPool(self.conv_block3(x))
                    x = torch.flatten(x, 1)
                    x = self.fc_block(x)

                    return x


################################################################
########################### Training ###########################
################################################################


def training(session_state):
    st.write("""## 5. Training""")

    st.write(
        """Während des Trainingsprozesses werden dem im vorherigen Unterkapitel *Netzwerkarchitektur* beschriebenen Modell **Trainingsbilder** mit bekannten **Grundwahrheiten** übergeben.
        Das **Modell** trifft zu jedem Eingabebild eine **Vorhersage**, die dann über die **Lossfunktion** mit der Grundwahrheit verglichen wird.
        Basierend auf der Ausgabe der Lossfunktion werden durch die **Backpropagation** die **Parameter** des Modells angepasst.
        Dieser Prozess wird so lange wiederholt, bis die Parameter des Modells so weit angepasst sind, dass sie bei der Backpropagation nicht mehr geändert werden müssen."""
    )
    st.image("StreamlitApps/Lektion-4/l2t_imgs/trainloop.png", caption="Trainings-Loop")

    st.markdown("---")

    st.write("**Für Coding-interessierte Lernende:**")
    if st.checkbox("""Gesamten Code für Training anzeigen""", False):
        st.write(
            "*Code für Einrichten von Loss-, und Optimierungs-Funktion sowie Einstellung von Batch-Größe und Epochenzahl:*"
        )

        st.code(
            """model = classificationCNN().cuda()

class_weight = torch.sqrt(1.0/(torch.bincount(data_label[idx_train].view(-1)).float()))
class_weight = class_weight/class_weight.mean()

loss_function = nn.CrossEntropyLoss(weight=class_weight.cuda())

learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 25
n_epochs = 26
"""
        )

        st.write(
            "*Code für den Trainings-Loop inklusive Verwendung von Validierungsdaten:*"
        )

        st.code(
            """

losses_training = []
losses_validation = []

best_loss = np.infty
bestNet = classificationCNN().cuda()

t0 = time.time()
for epoch in range(n_epochs):

    ########################################
    #               TRAINING               #
    ########################################

    sum_loss = 0

    train_batches = torch.randperm(len(idx_train))[:len(idx_train)-len(idx_train)%batch_size].view(-1,batch_size)

    # Parameters must be trainable
    model.train()
    with torch.set_grad_enabled(True):

        # main loop to process all training samples (packed into batches)
        for batch_idx in train_batches:
            source_image = data_img[batch_idx].cuda()
            target_label = data_label[batch_idx].cuda()

            # forward run
            predicted_label = model(source_image)

            #compute loss
            loss = loss_function(predicted_label, target_label)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        losses_training.append(sum_loss/ len(train_batches))

    ########################################
    #              VALIDATION              #
    ########################################

    sum_loss = 0

    # Parameters must not be trainable
    model.eval()
    with torch.set_grad_enabled(False):

        val_batches = torch.randperm(len(idx_val))[:len(idx_val)-len(idx_val)%batch_size_val].view(-1,batch_size_val)


        # main loop to process all validation samples (packed into batches)
        for batch_idx in val_batches:
            source_image = data_img[batch_idx].cuda()
            target_label = data_label[batch_idx].cuda()
            predicted_label = model(source_image) # forward pass

            loss = loss_function(predicted_label, target_label)  # compute loss

            # No need to backpropagate here!
            sum_loss += loss.item() / len(val_batches)

        validation_loss = sum_loss
        losses_validation.append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state_dict = bestNet.state_dict()
            state_dict = model.state_dict()
            for name, param in state_dict.items():
                best_state_dict[name].copy_(param)

    # scheduler will adapt the learning rate once the step count reaches threshold
    scheduler.step()

    if epoch % every_epoch == 0:
        t1 = time.time()
        print("{:.2f}".format(t1-t0), 's --- Epoch ' + str(epoch) +': Training loss ' + str(losses_training[-1]) + ', Validation loss ' + str(losses_validation[-1]))

print('Finished Training')
print('Best validation loss: ' + str(best_loss))
"""
        )


##################################################################
########################### Evaluation ###########################
##################################################################


def evaluation(session_state):
    st.write("""## 6. Evaluation""")

    accuracy_latex = r"""
$$
ACCURACY = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}
$$
"""

    st.markdown(
        """Im vorherigen Unterkapitel *Training* wurde das Modell `bestNet` abgespeichert, das auf den Validierungsdaten die besten Ergebnisse erzielt hat.
        Dieses Modell soll jetzt während der sogenannten **Inferenz** auf die Testdaten angewendet werden, um zu evaluieren, wie gut das Modell auf Bildern klassifizieren kann, die ihm während der Trainingsphase noch nicht präsentiert wurden."""
    )
    st.write(
        "Während der Evaluation wird bestimmt, bei wie vielen Fällen in den Testdaten "
    )
    st.markdown(
        "- korrekterweise eine Pneumonie erkannt wurde (*True Positive*, **TP** )"
    )
    st.markdown(
        "- korrekterweise *keine* Pneumonie erkannt wurde (*True Negative*, **TN**)"
    )
    st.markdown(
        "- fälschlicherweise eine Pneumonie erkannt wurde (*False Positive*, **FP**)"
    )
    st.markdown(
        "- fälschlicherweise *keine* Pneumonie erkannt wurde (*False Negative*, **FN**)."
    )
    st.write(
        "Die **Accuracy** gibt an, welcher Anteil an Bildern insgesamt richtig klassifiziert wurden:",
        accuracy_latex,
    )

    st.markdown("---")

    st.markdown("""Bei Starten der Inferenz werden""")
    st.markdown("(1) die Testbilder und Grundwahrheiten geladen")
    st.markdown("(2) das gespeicherte Modell `bestNet`auf die Testbilder angewendet")
    st.markdown("(3) die Ausgaben des Modells mit den Grundwahrheiten verglichen")

    if st.checkbox("""Inferenz starten""", False):
        data_img = load_data()
        data_label = load_data_label()
        idx_test = load_idx_test()

        model = classificationCNN()
        model = load_model()

        bestNet = model

        with st.echo():
            bestNet.eval()
            y_gt = []
            y_pred = []

            for ind in idx_test:

                # (1) load test images and ground truth label
                source_image = data_img[ind].unsqueeze(0)
                target_label = data_label[ind]

                # (2) apply model
                classification = bestNet(source_image).argmax(1)

                y_gt.append(target_label.item())
                y_pred.append(classification.item())

            # (3) calculate TP, TN, FP, FN and accuracy
            tn, fp, fn, tp = confusion_matrix(y_gt, y_pred).ravel()
            acc = (tp + tn) / len(idx_test)

            st.write("Scores (Testdatensatz):")
            st.write("TP: ", tp, "TN: ", tn, "FP: ", fp, "FN: ", fn)
            st.write("ACCURACY = ", acc)


#####################################################################
########################### Visualisation ###########################
#####################################################################


def visualisation(session_state):
    st.write("""## 7. Ergebnisvisualisierungen""")

    data_img = load_data()
    result_list = torch.load("StreamlitApps/Lektion-4/result_list.pth")

    st.write(
        """Hier wird visualisiert, welche Röntgenbilder richtig und welche Röntgenbilder aus dem Testdatensatz falsch klassifiziert wurden. """
    )
    st.write(
        "Insgesamt wurden",
        len(result_list[0]),
        " Bilder korrekt klassifiziert und ",
        len(result_list[1]),
        " falsch klassifiziert.",
    )
    st.write(
        """In der Bildüberschrift wird jeweils angegeben, ob es sich bei dem angezeigten Fall um TP (*True Positive*), TN (*True Negative*), FP (*False Positive*) oder FN (*False Negative*) handelt."""
    )

    col1, col2 = st.columns(2)
    col1.write("**Korrekt klassifizierte Testbilder**")
    idx = col1.slider("Bildindex", 0, len(result_list[0]) - 1, 0)
    img_ind, result_corr = result_list[0][idx]
    corr_img = data_img[img_ind, 0]

    col2.write("**Falsch klassifizierte Testbilder**")
    idx = col2.slider("Bildindex", 0, len(result_list[1]) - 1, 0)
    img_ind, result_mis = result_list[1][idx]
    mis_img = data_img[img_ind, 0]

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(corr_img, "gray")
    axes[0].axis("off")
    axes[0].set_title(result_corr)
    axes[1].imshow(mis_img, "gray")
    axes[1].axis("off")
    axes[1].set_title(result_mis)
    plt.subplots_adjust(hspace=-0.3, wspace=0.1)
    st.pyplot(fig)


#################################################################
########################### Main Page ###########################
#################################################################


session_state = st.session_state
if "button_id" not in session_state:
    session_state["button_id"] = ""
if "slider_value" not in session_state:
    session_state["slider_value"] = 0
st.set_page_config(page_title="KI Campus: Learn2Trust - Lektion 3", page_icon=":pencil2:")
st.title("KI Campus: Learn2Trust ")
st.write("## **Lektion 4: Klassifizierung**")

st.sidebar.subheader("Unterkapitel")
PAGES = {
    "1. Einführung": intro,
    "2. Datensatz": dataset,
    "3. Labelquiz": labelquiz,
    "4. Netzwerkarchitektur": architecture,
    "5. Training": training,
    "6. Evaluation": evaluation,
    "7. Ergebnisvisualisierung": visualisation,
}
page = st.sidebar.selectbox("Auswahl:", options=list(PAGES.keys()))

st.sidebar.markdown("""---""")

st.sidebar.write('Dieses Projekt wird bereitgestellt auf der')
link = "[KI-Campus Website](https://ki-campus.org/)"
st.sidebar.markdown(link, unsafe_allow_html=True)
st.sidebar.image("StreamlitApps/Lektion-4/l2t_imgs/KICampusLogo.png", use_column_width=True)

PAGES[page](session_state)
