(define (problem tireworld-067-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l7 l8 l9 - location)
    (:init (road l0 l34) (road l0 l51) (road l1 l42) (road l1 l51) (road l10 l52) (road l10 l59) (road l11 l43) (road l11 l47) (road l12 l46) (road l12 l5) (road l13 l21) (road l13 l58) (road l14 l4) (road l14 l43) (road l15 l35) (road l15 l52) (road l16 l38) (road l16 l62) (road l17 l20) (road l17 l65) (road l18 l36) (road l18 l9) (road l19 l30) (road l19 l64) (road l2 l33) (road l2 l6) (road l20 l17) (road l20 l31) (road l21 l13) (road l21 l64) (road l22 l47) (road l23 l62) (road l23 l63) (road l24 l30) (road l24 l50) (road l25 l37) (road l25 l55) (road l26 l3) (road l26 l39) (road l27 l28) (road l27 l57) (road l28 l27) (road l28 l39) (road l29 l45) (road l29 l66) (road l3 l26) (road l3 l31) (road l30 l19) (road l30 l24) (road l31 l20) (road l31 l3) (road l32 l50) (road l32 l6) (road l33 l2) (road l33 l40) (road l34 l0) (road l34 l7) (road l35 l15) (road l35 l49) (road l36 l18) (road l36 l61) (road l37 l25) (road l37 l56) (road l38 l16) (road l38 l5) (road l39 l26) (road l39 l28) (road l39 l44) (road l4 l14) (road l4 l8) (road l40 l33) (road l40 l46) (road l41 l57) (road l41 l58) (road l42 l1) (road l42 l45) (road l43 l11) (road l43 l14) (road l43 l53) (road l44 l39) (road l44 l54) (road l45 l29) (road l45 l42) (road l46 l12) (road l46 l40) (road l47 l11) (road l47 l22) (road l47 l7) (road l48 l62) (road l48 l8) (road l49 l35) (road l49 l65) (road l5 l12) (road l5 l38) (road l50 l24) (road l50 l32) (road l51 l0) (road l51 l1) (road l52 l10) (road l52 l15) (road l53 l43) (road l54 l44) (road l54 l56) (road l55 l25) (road l55 l60) (road l56 l37) (road l56 l54) (road l57 l27) (road l57 l41) (road l58 l13) (road l58 l41) (road l59 l10) (road l59 l63) (road l6 l2) (road l6 l32) (road l60 l55) (road l60 l9) (road l61 l36) (road l61 l66) (road l62 l16) (road l62 l23) (road l62 l48) (road l63 l23) (road l63 l59) (road l64 l19) (road l64 l21) (road l65 l17) (road l65 l49) (road l66 l29) (road l66 l61) (road l7 l34) (road l7 l47) (road l8 l4) (road l8 l48) (road l9 l18) (road l9 l60) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l17) (spare-in l2) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l43) (spare-in l47) (spare-in l48) (spare-in l50) (spare-in l52) (spare-in l55) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l8) (vehicle-at l53))
    (:goal (vehicle-at l22))
)