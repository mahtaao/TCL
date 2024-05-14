"""Real Data loading"""





import numpy as np
import torch
import os
import mne
import sys
# sys.path.append('C:\\Users\\mahta\\OneDrive\\Documents\\Work\\Codes\\IFT6168\\Final_project\\TCL')
from subfunc.showdata import *

# =============================================================
# =============================================================

def load_EEG_data(data_path = 'data/', 
                  num_segment = 10,
                  num_segmentdata = 100,
                  random_seed = 0):
    """
    Load EEG data from files and prepare edge data.
    Args:
        num_comp: dimension of source data determined by atlas
        num_segment: number of segments
        num_segmentdata: number of data-points in each segment (epochs)
        num_layer: number of layers of mixing-MLP
        random_seed: random seed
        participant_ids: list of participant IDs
        data_path: path to the data directory
        freq : frequency of the data
    Returns:
        sensor: observed signals [num_comp, num_data]
        source: source signals [num_comp, num_data]
        labels: labels [num_data], #segment
    """
    recording_dir = os.path.join(data_path, 'preprocessed')
    
    participant_ids = [f for f in os.listdir(recording_dir) if os.path.isdir(os.path.join(recording_dir, f))]
    print('Participant IDs:', participant_ids)
    
    sensor_data = []
    source_data = []

    for participant_id in participant_ids:
        # catch error if the participant inv or recording doesn't exist
        try:
            inv_path = os.path.join(data_path, 'template-mri', f'{participant_id}', 'vol_INV_OPT.txt')
            recording_path = os.path.join(data_path, 'preprocessed', f'{participant_id}', 'RestingState_epo.fif')
        
        except FileNotFoundError:
            print(f'Participant {participant_id} does not have both source and recording')
            continue
        
        parcellation =  np.loadtxt(os.path.join(data_path, 'vol_PARC_MTRX.txt'))
        print('parcellation shape:', parcellation.shape)

        inv = np.loadtxt(inv_path)
        recording = mne.read_epochs(recording_path)
        print('INFO: ', recording.info)
        # exclude bad channels
        recording = recording.drop_channels(recording.info['bads'])        
         
        
        # cut time slice for memory efficiency
        start_time = 15  # select a 10-second window starting from 10 seconds
        end_time = 20  # select a 10-second window starting from 10 seconds
        rec_time_length = end_time - start_time 
        freq = recording.info['sfreq']
        resampled_recording = recording.get_data().astype(np.float32)
        print('Recording data shape -1:', resampled_recording.shape)        
        # only use 5 events
        resampled_recording = resampled_recording[:rec_time_length, :, :]
        print('Recording data shape 0:', resampled_recording.shape) #(127, 501, 5)
        # reshape to #channels x #timepoints(i.e. fr x secods) 
        resampled_recording = np.transpose(resampled_recording, (1, 0, 2))  
        # reshape to (#channels, #timepoints x #fr)
        resampled_recording = resampled_recording.reshape(resampled_recording.shape[0], -1)
        print('Recording data shape 2:', resampled_recording.shape) #(210, 127, 501)
        

        
        # recording_data = recording.get_data()
        # recording_data = recording_data.transpose(0, 2, 1)  # reshape to (210, 501, 127)
        # source_activity = np.dot(recording_data, inv.T)  # dot product with inv transpose
        #  casting the recording_data and inv arrays to float32
        #   instead of float64. This can reduce the memory required   
        recording_data = recording.get_data(start_time, end_time).astype(np.float32)
        print('Recording data shape:', recording_data.shape) #(210, 501, 127)
        # FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning
        inv = inv.astype(np.float32)
        # recording_data = recording_data.transpose(0, 2, 1)

        # source_activity = np.dot(recording_data, inv.T)
        # sensor_data.append(recording)
        # source_data.append(source_activity)
        
        
        # Perform the dot product in chunks:
        chunk_size = parcellation.shape[1]
        source_activity = []
        for i in range(0, len(resampled_recording), chunk_size):
            print('i:', i)
            print('All the shapes: resampled_recording:', resampled_recording.shape, 'inv:', inv.T.shape, 'parcellation:', parcellation.T.shape)
            # resampled_recording: (1, 501) inv: (127, 67748) parcellation: (67748, 116)
            parcelated_inversed = np.dot(inv.T, parcellation.T)
            print('parcelated_inversed shape:', parcelated_inversed.shape)
            print('resampled_recording shape:', resampled_recording[:, i:i + chunk_size].shape)
            source_activity.append(np.dot(resampled_recording[:, i:i + chunk_size].T, parcelated_inversed).T)
            
        
        print('source_activity shape 1 :', np.array(source_activity).shape)
        source_activity = np.concatenate(source_activity, axis=0)
        print('source_activity shape 2 :', source_activity.shape)

    source = torch.from_numpy(np.array(source_activity))
    # Important shortcoming of TCL: num_sources are not necessarily equal to num_channels
    # in real world. We might have no idea how many sources are involved,
    # all we have access to is the number of channels (of the sensors we put).
    # but here we have assumed these are equal.
    # So, if there are more that 116 (# of parcels) channels, we will get rid of the extra channels
    recording = recording.pick(range(parcellation.shape[0]))
    recording_data = recording.get_data(start_time, end_time).astype(np.float32)
    print('Recording data shape 3:', resampled_recording.shape) #(127, 2505)
    # Generate labels
    num_data = resampled_recording.shape[1]
    num_segmentdata = int(freq * 1) # 1 second segments
    num_segment = int(num_data / num_segmentdata)
    print('num_segment:', num_segment, 'num_segmentdata:', num_segmentdata, 'num_data:', num_data)
    label = torch.zeros(num_data)
    for sn in range(num_segment):
        start_idx = num_segmentdata*sn
        end_idx = num_segmentdata*(sn+1)
        label[start_idx:end_idx] = sn
        print('start_idx:', start_idx, 'end_idx:', end_idx, sn)
        
    sensor = torch.from_numpy(np.array(resampled_recording[:parcellation.shape[0]]))

    print('final dataset informatin:', sensor.shape, source.shape, label.shape,
            'num_data:', num_data,
            'num_segment:', num_segment,
            'num_segmentdata:', num_segmentdata,
            'label', (label), 'label shape:', label.shape)
            
    return sensor, source, label


def generate_artificial_data(num_comp,
                             num_segment,
                             num_segmentdata,
                             num_layer,
                             num_segmentdata_test=None,
                             random_seed=0):
    """Generate artificial data.
    Args:
        num_comp: number of components
        num_segment: number of segments
        num_segmentdata: number of data-points in each segment
        num_layer: number of layers of mixing-MLP
        num_segmentdata_test: (option) number of data-points in each segment (testing data, if necessary)
        random_seed: (option) random seed
    Returns:
        sensor: observed signals [num_comp, num_data]
        source: source signals [num_comp, num_data]
        label: labels [num_data]
    """

    # Generate source signal
    source, label, L = gen_source_tcl(num_comp, num_segment, num_segmentdata, random_seed=random_seed)

    # Apply mixing MLP
    sensor, mixlayer = apply_MLP_to_source(source, num_layer, num_segment=num_segment, random_seed=random_seed)

    # Add test data (option)
    if num_segmentdata_test is not None:
        source_test, label_test, L_test = gen_source_tcl(num_comp, num_segment, num_segmentdata_test,
                                                         L=L, random_seed=random_seed+1) # change random_seed
        # Use same parameters of the training data
        sensor_test, mixlayer_test = apply_MLP_to_source(source_test, num_layer,
                                                         num_segment=num_segment, random_seed=random_seed)

    return sensor, source, label


# =============================================================
# =============================================================
def gen_source_tcl(num_comp,
                   num_segment,
                   num_segmentdata,
                   L=None,
                   Ltype='uniform',
                   Lrange=None,
                   sourcetype='laplace',
                   random_seed=0):
    """Generate source signal for TCL.
    Args:
        num_comp: number of components
        num_segment: number of segments
        num_segmentdata: number of data-points in each segment
        L: (option) modulation parameter. If not given, newly generate based on Ltype
        Ltype: (option) generation method of modulation parameter
        Lrange: (option) range of modulation parameter
        sourcetype: (option) Distribution type of source signal
        random_seed: (option) random seed
    Returns:
        source: source signals. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
        L: modulation parameter of each component/segment. 2D ndarray [num_comp, num_segment]
    """
    if Lrange is None:
        Lrange = [0, 1]
    print("Generating source...")

    # Initialize random generator
    np.random.seed(random_seed)

    # Change random_seed based on num_segment
    for i in range(num_segment):
        np.random.rand()

    # Generate modulation parameter
    if L is None:
        if Ltype == "uniform":
            L = np.random.uniform(min(Lrange), max(Lrange), [num_comp, num_segment])
        else:
            raise ValueError

    # Generate source signal ----------------------------------
    num_data = num_segment * num_segmentdata
    source = np.zeros([num_comp,num_data])
    label = np.zeros(num_data)
    sourcestd = np.zeros([num_comp,num_segment]) # For std check
    sourcemean = np.zeros([num_comp,num_segment]) # For mean check

    for sn in range(num_segment): # Segment
        for cn in range(num_comp): # Component

            start_idx = num_segmentdata*sn
            end_idx = num_segmentdata*(sn+1)

            if sourcetype == 'laplace':
                source_sc = np.random.laplace(0, 1 / np.sqrt(2), [1, num_segmentdata])
                source_sc = source_sc * L[cn,sn]
            else:
                raise ValueError

            source[cn,start_idx:end_idx] = source_sc

            # For check
            sourcestd[cn,sn] = np.std(source_sc)
            sourcemean[cn,sn] = np.mean(source_sc)

        # Label
        label[start_idx:end_idx] = sn

    return source, label, L


# =============================================================
# =============================================================
def apply_MLP_to_source(source,
                        num_layer,
                        num_segment = None,
                        iter4condthresh = 10000,
                        cond_thresh_ratio = 0.25,
                        layer_name_base = 'ip',
                        save_layer_data = False,
                        Arange=None,
                        nonlinear_type = 'ReLU',
                        negative_slope = 0.2,
                        random_seed=0):
    """Generate MLP and Apply it to source signal.
    Args:
        source: source signals. 2D ndarray [num_comp, num_data]
        num_layer: number of layers
        num_segment: (option) number of segments (only used to modulate random_seed)
        iter4condthresh: (option) number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: (option) percentile of condition number to decide its threshold
        layer_name_base: (option) layer name
        save_layer_data: (option) if true, save activities of all layers
        Arange: (option) range of value of mixing matrices
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
        random_seed: (option) random seed
    Returns:
        mixedsig: sensor signals. 2D ndarray [num_comp, num_data]
        mixlayer: parameters of mixing layers
    """
    if Arange is None:
        Arange = [-1, 1]
    print("Generating sensor signal...")

    # Subfuction to normalize mixing matrix
    def l2normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat*Amat,axis))
        Amat = Amat / l2norm
        return Amat

    # Initialize random generator
    np.random.seed(random_seed)
    # To change random_seed based on num_layer and num_segment
    for i in range(num_layer):
        np.random.rand()
    if num_segment is not None:
        for i in range(num_segment):
            np.random.rand()

    num_comp = source.shape[0]

    # Determine condThresh ------------------------------------
    condList = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        A = np.random.uniform(Arange[0],Arange[1],[num_comp,num_comp])
        A = l2normalize(A, axis=0)
        condList[i] = np.linalg.cond(A)

    condList.sort() # Ascending order
    condThresh = condList[int(iter4condthresh * cond_thresh_ratio)]
    print("    cond thresh: {0:f}".format(condThresh))

    # Generate mixed signal -----------------------------------
    mixedsig = source.copy()
    mixlayer = []
    for ln in range(num_layer-1,-1,-1):

        # Apply nonlinearity ----------------------------------
        if ln < num_layer-1: # No nolinearity for the first layer (source signal)
            if nonlinear_type == "ReLU": # Leaky-ReLU
                mixedsig[mixedsig<0] = negative_slope * mixedsig[mixedsig<0]
            else:
                raise ValueError

        # Generate mixing matrix ------------------------------
        condA = condThresh + 1
        while condA > condThresh:
            A = np.random.uniform(Arange[0], Arange[1], [num_comp, num_comp])
            A = l2normalize(A)  # Normalize (column)
            condA = np.linalg.cond(A)
            print("    L{0:d}: cond={1:f}".format(ln, condA))
        # Bias
        b = np.zeros([num_comp]).reshape([1,-1]).T

        # Apply bias and mixing matrix ------------------------
        mixedsig = mixedsig + b
        mixedsig = np.dot(A, mixedsig)

        # Storege ---------------------------------------------
        layername = layer_name_base + str(ln+1)
        mixlayer.append({"name":layername, "A":A.copy(), "b":b.copy()})
        # Storege data
        if save_layer_data:
            mixlayer[-1]["x"] = mixedsig.copy()

    return mixedsig, mixlayer


