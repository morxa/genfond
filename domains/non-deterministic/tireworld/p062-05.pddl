(define (problem tireworld-062-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l7 l8 l9 - location)
    (:init (road l0 l56) (road l0 l6) (road l1 l46) (road l1 l57) (road l10 l24) (road l10 l9) (road l11 l2) (road l11 l45) (road l12 l13) (road l12 l49) (road l13 l12) (road l13 l33) (road l14 l27) (road l14 l28) (road l15 l36) (road l15 l37) (road l16 l4) (road l16 l58) (road l17 l24) (road l17 l47) (road l18 l39) (road l18 l54) (road l19 l27) (road l19 l9) (road l2 l11) (road l2 l41) (road l20 l32) (road l20 l52) (road l21 l51) (road l21 l7) (road l22 l31) (road l22 l38) (road l22 l55) (road l23 l26) (road l23 l38) (road l24 l10) (road l24 l17) (road l25 l34) (road l25 l40) (road l26 l23) (road l26 l28) (road l27 l14) (road l27 l19) (road l27 l48) (road l28 l14) (road l28 l26) (road l29 l33) (road l29 l59) (road l3 l41) (road l3 l50) (road l30 l54) (road l31 l22) (road l31 l60) (road l32 l20) (road l32 l58) (road l33 l13) (road l33 l29) (road l34 l25) (road l34 l52) (road l35 l47) (road l35 l53) (road l36 l15) (road l36 l60) (road l37 l15) (road l37 l51) (road l37 l54) (road l38 l22) (road l38 l23) (road l38 l5) (road l39 l18) (road l39 l43) (road l39 l45) (road l4 l16) (road l4 l7) (road l40 l25) (road l40 l50) (road l41 l2) (road l41 l3) (road l41 l46) (road l41 l50) (road l42 l54) (road l42 l59) (road l43 l39) (road l43 l49) (road l44 l56) (road l44 l61) (road l45 l11) (road l45 l39) (road l46 l1) (road l46 l41) (road l47 l17) (road l47 l35) (road l48 l27) (road l49 l12) (road l49 l43) (road l5 l38) (road l5 l6) (road l50 l3) (road l50 l40) (road l50 l41) (road l51 l21) (road l51 l37) (road l52 l20) (road l52 l34) (road l52 l8) (road l53 l35) (road l53 l57) (road l54 l18) (road l54 l30) (road l54 l37) (road l54 l42) (road l54 l61) (road l54 l9) (road l55 l22) (road l55 l8) (road l56 l0) (road l56 l44) (road l57 l1) (road l57 l53) (road l58 l16) (road l58 l32) (road l59 l29) (road l59 l42) (road l6 l0) (road l6 l5) (road l60 l31) (road l60 l36) (road l61 l44) (road l61 l54) (road l7 l21) (road l7 l4) (road l8 l52) (road l8 l55) (road l9 l10) (road l9 l19) (road l9 l54) (spare-in l1) (spare-in l14) (spare-in l19) (spare-in l27) (spare-in l54) (spare-in l8) (spare-in l9) (vehicle-at l48))
    (:goal (vehicle-at l30))
)