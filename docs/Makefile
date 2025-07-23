SPHINXOPTS    = -q
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

.PHONY: html clean
html:
    # 这一行必须以一个 Tab 键开始，而不是空格
    $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

clean:
    # 这一行也必须以一个 Tab 键开始
    rm -rf $(BUILDDIR)