(define (problem tireworld-060-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l7 l8 l9 - location)
    (:init (road l0 l28) (road l1 l34) (road l1 l49) (road l10 l25) (road l10 l38) (road l11 l48) (road l12 l20) (road l13 l3) (road l13 l51) (road l14 l9) (road l15 l33) (road l16 l42) (road l17 l46) (road l18 l39) (road l19 l5) (road l2 l53) (road l20 l17) (road l21 l6) (road l22 l23) (road l23 l55) (road l24 l12) (road l25 l10) (road l26 l59) (road l27 l7) (road l28 l0) (road l28 l54) (road l29 l52) (road l29 l57) (road l3 l21) (road l30 l16) (road l31 l25) (road l32 l27) (road l33 l11) (road l34 l1) (road l35 l44) (road l36 l24) (road l37 l51) (road l37 l54) (road l38 l57) (road l39 l18) (road l39 l2) (road l4 l56) (road l40 l22) (road l41 l26) (road l41 l7) (road l42 l19) (road l43 l15) (road l44 l34) (road l45 l30) (road l45 l48) (road l46 l17) (road l46 l8) (road l47 l32) (road l48 l45) (road l49 l50) (road l5 l19) (road l5 l55) (road l5 l58) (road l50 l18) (road l50 l47) (road l51 l13) (road l52 l14) (road l53 l2) (road l53 l38) (road l54 l28) (road l54 l37) (road l55 l5) (road l56 l31) (road l57 l29) (road l57 l38) (road l58 l0) (road l58 l5) (road l59 l4) (road l6 l21) (road l6 l36) (road l7 l41) (road l8 l35) (road l8 l46) (road l9 l14) (road l9 l43) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l43) (spare-in l44) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l9) (vehicle-at l40))
    (:goal (vehicle-at l0))
)