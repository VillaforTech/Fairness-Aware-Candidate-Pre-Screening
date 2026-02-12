"""Allow running governance gate as python -m fairness_project.governance."""

import sys

from fairness_project.governance.gate import main

sys.exit(main())
