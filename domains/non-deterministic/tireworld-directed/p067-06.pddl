(define (problem tireworld-067-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l7 l8 l9 - location)
    (:init (road l0 l45) (road l0 l49) (road l1 l21) (road l1 l26) (road l10 l3) (road l10 l44) (road l11 l56) (road l12 l29) (road l12 l58) (road l13 l43) (road l13 l9) (road l14 l21) (road l14 l41) (road l14 l55) (road l15 l28) (road l16 l43) (road l17 l24) (road l17 l49) (road l18 l57) (road l18 l61) (road l19 l59) (road l2 l41) (road l2 l48) (road l2 l55) (road l20 l23) (road l20 l54) (road l21 l1) (road l21 l14) (road l22 l56) (road l22 l57) (road l23 l20) (road l23 l24) (road l23 l38) (road l24 l17) (road l25 l50) (road l26 l1) (road l26 l62) (road l27 l3) (road l27 l5) (road l28 l15) (road l28 l38) (road l29 l12) (road l29 l59) (road l3 l10) (road l3 l27) (road l30 l4) (road l31 l51) (road l31 l60) (road l32 l34) (road l32 l8) (road l33 l35) (road l33 l39) (road l34 l32) (road l34 l9) (road l35 l33) (road l35 l5) (road l36 l30) (road l37 l19) (road l37 l53) (road l38 l23) (road l38 l28) (road l39 l33) (road l4 l30) (road l4 l42) (road l40 l36) (road l41 l14) (road l41 l2) (road l42 l4) (road l42 l51) (road l43 l13) (road l44 l10) (road l44 l15) (road l45 l40) (road l46 l58) (road l46 l7) (road l47 l59) (road l48 l66) (road l49 l0) (road l49 l17) (road l5 l27) (road l5 l35) (road l50 l11) (road l51 l31) (road l51 l42) (road l52 l47) (road l52 l60) (road l53 l37) (road l53 l8) (road l54 l20) (road l54 l66) (road l55 l14) (road l55 l2) (road l56 l11) (road l56 l22) (road l56 l65) (road l57 l18) (road l57 l22) (road l58 l12) (road l58 l46) (road l59 l19) (road l59 l29) (road l6 l10) (road l6 l65) (road l60 l52) (road l60 l6) (road l60 l64) (road l61 l18) (road l61 l63) (road l62 l26) (road l62 l64) (road l63 l61) (road l64 l62) (road l65 l56) (road l65 l6) (road l66 l25) (road l7 l39) (road l7 l46) (road l8 l53) (road l9 l34) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l5) (spare-in l50) (spare-in l53) (spare-in l54) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l66) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l16))
    (:goal (vehicle-at l63))
)