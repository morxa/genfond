(define (problem tireworld-051-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l1 l39) (road l10 l21) (road l11 l14) (road l12 l0) (road l12 l16) (road l13 l18) (road l14 l20) (road l15 l29) (road l16 l40) (road l17 l2) (road l18 l13) (road l18 l4) (road l19 l31) (road l2 l43) (road l20 l8) (road l21 l47) (road l22 l24) (road l22 l46) (road l23 l42) (road l24 l22) (road l24 l9) (road l25 l32) (road l26 l0) (road l27 l23) (road l27 l47) (road l28 l13) (road l29 l48) (road l3 l15) (road l30 l44) (road l31 l19) (road l31 l43) (road l32 l17) (road l32 l25) (road l33 l30) (road l34 l25) (road l34 l41) (road l35 l34) (road l36 l35) (road l36 l43) (road l37 l49) (road l38 l10) (road l4 l1) (road l40 l37) (road l41 l9) (road l42 l19) (road l43 l36) (road l44 l50) (road l45 l33) (road l46 l28) (road l47 l27) (road l48 l45) (road l49 l7) (road l5 l11) (road l5 l50) (road l50 l5) (road l6 l26) (road l7 l38) (road l8 l6) (road l9 l24) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l39))
)