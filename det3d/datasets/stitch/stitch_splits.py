
train_detect = \
    ['scene_060', 'scene_057', 'scene_045', 'scene_041','scene_037',
     'scene_034', 'scene_068', 'scene_029', 'scene_082', 'scene_009',
     'scene_030', 'scene_010', 'scene_055', 'scene_054', 'scene_059',
     'scene_044', 'scene_050', 'scene_061', 'scene_066', 'scene_072',
     'scene_043', 'scene_006', 'scene_002', 'scene_046', 'scene_016',
     'scene_026', 'scene_031', 'scene_063', 'scene_021', 'scene_074',
     'scene_042', 'scene_011', 'scene_033', 'scene_005', 'scene_032',
     'scene_013', 'scene_081', 'scene_056', 'scene_001', 'scene_069',
     'scene_004', 'scene_049', 'scene_073', 'scene_023', 'scene_053', 
     'scene_079', 'scene_080', 'scene_058', 'scene_007', 'scene_020', 
     'scene_064', 'scene_017', 'scene_065', 'scene_077', 'scene_051',
     'scene_024', 'scene_018', 'scene_047', 'scene_027', 'scene_071',
     'scene_022', 'scene_015', 'scene_014', 'scene_019', 'scene_078',
     'scene_052']
train_track = \
    ['scene_001', 'scene 002', 'scene_004', 'scene_005', 'scene_006',
     'scene 007', 'scene_009', 'scene_010', 'scene_011', 'scene_013',
     'scene_014','scene_ 015', 'scene_016', 'scene_017', 'scene_018',
     'scene_019', 'scene_020']
train = list(sorted(set(train_detect + train_track)))

val = \
    ['scene_012', 'scene_067', 'scene_038', 'scene_035', 'scene_076',
     'scene_008', 'scene_048', 'scene_075', 'scene_025', 'scene_028',
     'scene_036', 'scene_039', 'scene_062', 'scene_070', 'scene_003',
     'scene_040']
val_track = \
    ['scene_003', 'scene_008', 'scene_012']
val = list(sorted(set(val + val_track)))

