import sys
from functools import reduce

import torch
from PIL import Image
from torch.nn.modules.module import _addindent


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram with note labels and a sidebar
    """
    from PIL import ImageDraw, ImageFont
    from .constants import MIN_MIDI

    # 1. Process the raw data into an image
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image_data = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    
    raw_image = Image.fromarray(image_data, 'RGB')
    width, height = raw_image.size
    
    # 2. Setup the Sidebar and Canvas
    sidebar_width = 60
    full_width = width + sidebar_width
    zoomed_height = height * zoom
    
    # Create a new white canvas
    final_image = Image.new('RGB', (full_width, zoomed_height), (255, 255, 255))
    
    # Paste the zoomed pianoroll
    zoomed_pianoroll = raw_image.resize((width, zoomed_height), Image.NEAREST)
    final_image.paste(zoomed_pianoroll, (sidebar_width, 0))
    
    # 3. Draw Labels and Grid Lines
    draw = ImageDraw.Draw(final_image)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i in range(height):
        midi = MIN_MIDI + i
        # Calculate Y position (flipped because bin 0 is at the bottom)
        y_pos = (height - 1 - i) * zoom
        
        # Draw a faint gray line for every C and E (guitar reference points)
        if midi % 12 in [0, 4]:
            draw.line([(sidebar_width, y_pos), (full_width, y_pos)], fill=(220, 220, 220))
        
        # Label every C and the lowest/highest notes
        if midi % 12 == 0 or i == 0 or i == height - 1:
            note_name = note_names[midi % 12]
            octave = (midi // 12) - 1
            label = f"{note_name}{octave} ({midi})"
            draw.text((5, y_pos), label, fill=(0, 0, 0))

    final_image.save(path)
