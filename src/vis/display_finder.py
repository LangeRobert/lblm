from screeninfo import get_monitors


def find_display():
    monitors = get_monitors()
    best_index = 0
    best_ratio = 0

    for i, m in enumerate(monitors):
        if m.width > 0:
            ratio = m.height / m.width
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = i

    print(f"Best display: {monitors[best_index].name} (index {best_index}) with ratio {best_ratio:.2f}")
    return best_index
