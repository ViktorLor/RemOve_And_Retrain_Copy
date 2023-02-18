from saliency_helper_by_vlo import generate_masks
from saliency_helper_by_vlo import apply_mask_to_image
import torch
# generate 3x224x224 random image

# define testtcase

gen = True


if gen:

    image = torch.rand(1, 3, 224, 224)
    # plot image
    import matplotlib.pyplot as plt
    plt.imshow(image[0].permute(1,2,0))
    plt.show()

    mask,i = generate_masks(image, [0.5])

    r = apply_mask_to_image(image.clone(), mask)

    # plot r
    import matplotlib.pyplot as plt
    plt.imshow(r.permute(1,2,0))
    plt.show()

    # count how many pixels are equal

    print(torch.sum(image[0] == r[0]))

#%%
