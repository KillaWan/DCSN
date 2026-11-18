Third-party code in the DCSN paper release
==========================================

The seven adapted files in dcsn/backbones are derived from models/ and
layers/ at https://github.com/Hank0626/DDN/tree/72b8d9c595ca81e70500919689f8715ed133e6d2
(MIT; DDN-LICENSE). Only forecasting backbones and their dependencies are
included; the DDN normalization model is not included.

SOURCE_REPORT.json identifies each file by its pinned upstream URL, upstream
SHA-256 and shipped SHA-256. Adaptations select the code paths used in the
paper, use package-local imports and omit unused upstream branches, logging
and allocations. The DLinear factory supplies the paper's moving average of
25. FEDformer uses the Fourier implementation.

Original backbone notices are retained: DLinear-LICENSE (Apache-2.0),
Autoformer-LICENSE (MIT) and FEDformer-LICENSE (MIT). The FAN-LICENSE
(Apache-2.0) is retained for data-loading and time-feature provenance,
not for a FAN forecasting or normalization implementation. The manifest
links to the exact LICENSE files for all five notices.

These licenses apply to their respective upstream contributions. The dataset
download manifest links to separate sources; the datasets are not bundled.
