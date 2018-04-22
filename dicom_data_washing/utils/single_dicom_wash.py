import pydicom
from pydicom.errors import InvalidDicomError


def single_dicom_wash(dataset: pydicom.Dataset):
    modality = dataset.get('Modality')
    if modality not in ['CT', 'PT']:
        raise InvalidDicomError("Unknown Modality: {}".format(modality))
    instance_number = dataset.get('InstanceNumber')
    output_filename = "{}_{:>03}".format(modality, instance_number) + '.dcm'
    dataset.__setattr__('PatientID', 'secret')
    dataset.__setattr__('PatientBirthDate', '19000101')
    dataset.__setattr__('PatientName', 'secret')
    dataset.__setattr__('PatientAge', '00Y')
    return dataset, modality, output_filename
