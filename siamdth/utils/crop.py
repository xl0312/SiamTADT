def crop_domain(im, c):
    radius = 64
    side = 128
    center_x = c[0]
    center_y = c[1]
    h, w = im.shape
    if h > 128 and w > 128:

        if center_x - radius < 0:
            fill_x = int(radius - center_x)
            right_x = fill_x + radius + center_x
            x1 = 0
            x2 = int(right_x)
            if x2 - x1 < side:
                x2 = x2 + 1

            if center_y - radius <= 0:
                fill_y = int(radius - center_y)
                up_y = int(fill_y + radius + center_y)

                y1 = 0
                y2 = up_y
                if y2 - y1 < side:y2 = y2 + 1
                crop = im[y1:y2, x1:x2]
                return crop
            if center_y + radius >= h:

                fill_y = int(center_y + radius - h)
                bottom_y = int(center_y - fill_y - radius)

                y1 = bottom_y
                y2 = h
                if y2 - y1 < side:y1 = y1 - 1
                crop = im[y1:y2, x1:x2]
                return crop
            if center_y - radius > 0 or center_y + radius < h:
                y1 = int(center_y - radius)
                y2 = int(center_y + radius)
                if y2 - y1 < side: y2 = y2 + 1
                crop = im[y1:y2, x1:x2]
                return crop
        if center_x + radius > w:
            fill_x = int(center_x + radius - w)
            left_x = int(center_x - fill_x - radius)
            x1 = left_x
            x2 = w
            if x2 - x1 < side: x1 = x1 - 1
            if center_y - radius <= 0:
                fill_y = int(radius - center_y)
                up_y = int(fill_y + radius + center_y)
                y1 = 0
                y2 = up_y
                if y2 - y1 < side: y2 = y2 + 1
                crop = im[y1:y2, x1:x2]
                return crop
            if center_y + radius >= h:
                fill_y = int(center_y + radius - h)
                bottom_y = int(center_y - fill_y - radius)
                y1 = bottom_y
                y2 = h
                if y2 - y1 < side: y1 = y1 - 1
                crop = im[y1:y2, x1:x2]
                return crop
            if center_y - radius > 0 or center_y + radius < h:
                y1 = int(center_y - radius)
                y2 = int(center_y + radius)
                if y2 - y1 < side: y2 = y2 + 1
                crop = im[y1:y2, x1:x2]
                return crop
        if center_x - radius > 0 and center_x + radius < w:

            x1 = int(center_x - radius)
            x2 = int(center_x + radius)
            if x2 - x1 < side: x2 = x2 + 1
            if center_y - radius < 0:
                fill_y = radius - center_y
                up_y = int(center_y + radius + fill_y)
                y1 = 0
                y2 = up_y
                if y2 - y1 < side: y2 = y2 + 1
                crop = im[y1:y2, x1:x2]
                return crop
            if center_y + radius > h:
                fill_y = center_y + radius - h
                bottom_y = int(center_y - radius - fill_y)
                y1 = bottom_y
                y2 = h
                if y2 - y1 < side: y1 = y1 - 1
                crop = im[y1:y2, x1:x2]

                return crop

        if center_x - radius > 0 and center_x + radius < w and center_y - radius > 0 and center_y + radius < h:
            x1 = int(center_x - radius)
            x2 = int(center_x + radius)
            y1 = int(center_y - radius)
            y2 = int(center_y + radius)
            if x2 - x1 < side: x2 = x2 + 1
            if y2 - y1 < side: y2 = y2 + 1
            crop = im[y1:y2, x1:x2]
            return crop
    else:
        crop =im
        return crop

