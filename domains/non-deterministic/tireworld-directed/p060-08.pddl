(define (problem tireworld-060-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l7 l8 l9 - location)
    (:init (road l0 l6) (road l1 l19) (road l10 l30) (road l11 l9) (road l12 l25) (road l12 l6) (road l13 l29) (road l14 l20) (road l15 l48) (road l16 l23) (road l17 l31) (road l18 l8) (road l19 l28) (road l19 l42) (road l2 l16) (road l2 l47) (road l20 l14) (road l20 l43) (road l21 l38) (road l22 l56) (road l23 l14) (road l24 l13) (road l25 l50) (road l26 l34) (road l27 l26) (road l28 l10) (road l29 l45) (road l3 l36) (road l30 l11) (road l31 l35) (road l31 l51) (road l32 l57) (road l33 l24) (road l34 l26) (road l34 l41) (road l35 l31) (road l35 l41) (road l36 l54) (road l37 l41) (road l38 l3) (road l39 l46) (road l40 l54) (road l40 l58) (road l41 l35) (road l41 l52) (road l42 l18) (road l43 l32) (road l44 l22) (road l45 l4) (road l46 l27) (road l47 l33) (road l48 l44) (road l49 l21) (road l5 l49) (road l50 l15) (road l51 l41) (road l52 l2) (road l53 l17) (road l54 l36) (road l54 l40) (road l55 l37) (road l56 l1) (road l57 l0) (road l58 l59) (road l59 l53) (road l6 l12) (road l6 l25) (road l7 l5) (road l8 l55) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l59) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l39))
    (:goal (vehicle-at l4))
)