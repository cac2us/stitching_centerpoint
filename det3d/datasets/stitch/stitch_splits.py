
train_detect = \
    ['scene_016', 'scene_008', 'scene_044', 'scene_021', 'scene_030', 
 'scene_020', 'scene_038', 'scene_036', 'scene_022', 'scene_041', 'scene_001',
 'scene_007', 'scene_029', 'scene_040', 'scene_024', 'scene_013', 'scene_009',
  'scene_012', 'scene_042', 'scene_019', 'scene_003', 'scene_025', 'scene_034',
   'scene_026', 'scene_028', 'scene_039', 'scene_017', 'scene_045', 'scene_046',
    'scene_023', 'scene_033', 'scene_043', 'scene_047', 'scene_018']
train_track = \
    ['scene_015',  'scene_048', 'scene_031', 'scene_006', 'scene_002']
train = list(sorted(set(train_detect + train_track)))

val = \
    ['scene_027', 'scene_010', 'scene_035', 'scene_014', 'scene_005', 'scene_037', 'scene_004']
val_track = \
    ['scene_011', 'scene_032']
val = list(sorted(set(val + val_track)))