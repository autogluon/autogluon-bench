G4_2 = "g4dn.2xlarge"
G4_4 = "g4dn.4xlarge"
G4_12 = "g4dn.12xlarge"
G5_4 = "g5.4xlarge"
P3_2 = "p3.2xlarge"

gpu_map = {
    G4_2: 1,
    G4_4: 1,
    G4_12: 4,
    G5_4: 1,
    P3_2: 1,
}

vcpu_map = {
    G4_2: 8,
    G4_4: 16,
    G4_12: 48,
    G5_4: 16,
    P3_2: 8,
}

memory_map = {
    G4_2: 32000,
    G4_4: 64000,
    G4_12: 192000,
    G5_4: 64000,
    P3_2: 61000,
}
