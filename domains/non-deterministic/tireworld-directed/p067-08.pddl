(define (problem tireworld-067-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l40) (road l0 l65) (road l1 l26) (road l1 l45) (road l10 l3) (road l10 l54) (road l11 l28) (road l11 l66) (road l12 l15) (road l13 l19) (road l13 l26) (road l14 l35) (road l14 l47) (road l15 l12) (road l15 l22) (road l16 l0) (road l16 l60) (road l17 l23) (road l18 l46) (road l18 l7) (road l19 l13) (road l19 l17) (road l19 l46) (road l2 l50) (road l2 l58) (road l20 l22) (road l20 l36) (road l21 l27) (road l22 l20) (road l23 l17) (road l23 l8) (road l24 l38) (road l25 l1) (road l26 l1) (road l26 l12) (road l26 l13) (road l27 l24) (road l27 l29) (road l27 l37) (road l28 l11) (road l29 l27) (road l29 l56) (road l3 l10) (road l3 l44) (road l3 l5) (road l30 l39) (road l30 l51) (road l31 l32) (road l31 l4) (road l31 l47) (road l32 l31) (road l32 l51) (road l33 l52) (road l33 l9) (road l34 l39) (road l35 l14) (road l35 l41) (road l36 l20) (road l36 l40) (road l37 l27) (road l38 l24) (road l38 l50) (road l39 l30) (road l39 l34) (road l4 l25) (road l4 l31) (road l40 l0) (road l40 l36) (road l41 l35) (road l42 l6) (road l42 l64) (road l43 l21) (road l43 l44) (road l44 l3) (road l44 l43) (road l45 l1) (road l46 l18) (road l46 l19) (road l47 l14) (road l47 l31) (road l48 l56) (road l48 l59) (road l49 l53) (road l5 l3) (road l50 l2) (road l50 l38) (road l51 l30) (road l51 l32) (road l52 l33) (road l52 l55) (road l53 l45) (road l53 l49) (road l54 l10) (road l55 l52) (road l55 l59) (road l56 l29) (road l56 l48) (road l56 l62) (road l57 l61) (road l57 l8) (road l58 l2) (road l58 l6) (road l59 l48) (road l59 l55) (road l6 l42) (road l6 l58) (road l60 l16) (road l60 l7) (road l61 l34) (road l61 l57) (road l62 l18) (road l62 l56) (road l63 l28) (road l63 l64) (road l64 l42) (road l64 l63) (road l65 l0) (road l65 l54) (road l66 l49) (road l7 l18) (road l8 l23) (road l8 l57) (road l9 l41) (spare-in l11) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l27) (spare-in l3) (spare-in l34) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l53) (spare-in l6) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l7) (vehicle-at l5))
    (:goal (vehicle-at l37))
)