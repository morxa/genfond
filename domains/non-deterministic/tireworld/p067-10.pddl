(define (problem tireworld-067-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l31) (road l1 l27) (road l1 l59) (road l10 l32) (road l10 l51) (road l11 l5) (road l11 l65) (road l12 l35) (road l12 l46) (road l13 l0) (road l13 l35) (road l14 l31) (road l14 l50) (road l15 l24) (road l15 l43) (road l16 l23) (road l16 l45) (road l17 l44) (road l17 l49) (road l18 l2) (road l18 l26) (road l18 l37) (road l19 l52) (road l19 l56) (road l2 l18) (road l2 l47) (road l2 l57) (road l2 l58) (road l20 l30) (road l20 l42) (road l20 l60) (road l21 l39) (road l21 l4) (road l22 l56) (road l22 l64) (road l23 l16) (road l23 l64) (road l24 l15) (road l24 l40) (road l24 l44) (road l25 l37) (road l25 l7) (road l26 l18) (road l26 l29) (road l27 l1) (road l27 l41) (road l28 l61) (road l28 l9) (road l29 l26) (road l29 l6) (road l3 l33) (road l3 l63) (road l30 l20) (road l30 l48) (road l31 l0) (road l31 l14) (road l32 l10) (road l32 l41) (road l33 l3) (road l33 l42) (road l33 l43) (road l34 l39) (road l34 l40) (road l35 l12) (road l35 l13) (road l36 l38) (road l36 l9) (road l37 l18) (road l37 l25) (road l38 l36) (road l38 l60) (road l39 l21) (road l39 l34) (road l4 l21) (road l4 l62) (road l40 l24) (road l40 l34) (road l41 l27) (road l41 l32) (road l42 l20) (road l42 l33) (road l43 l15) (road l43 l33) (road l44 l17) (road l44 l24) (road l45 l16) (road l45 l54) (road l46 l12) (road l46 l48) (road l47 l2) (road l47 l53) (road l48 l30) (road l48 l46) (road l49 l17) (road l49 l51) (road l5 l11) (road l5 l52) (road l50 l14) (road l50 l53) (road l51 l10) (road l51 l49) (road l52 l19) (road l52 l5) (road l53 l47) (road l53 l50) (road l54 l45) (road l54 l55) (road l55 l54) (road l55 l8) (road l56 l19) (road l56 l22) (road l57 l2) (road l57 l62) (road l58 l2) (road l58 l8) (road l59 l1) (road l59 l61) (road l6 l29) (road l6 l66) (road l60 l20) (road l60 l38) (road l61 l28) (road l61 l59) (road l62 l4) (road l62 l57) (road l63 l3) (road l63 l66) (road l64 l22) (road l64 l23) (road l65 l11) (road l65 l7) (road l66 l6) (road l66 l63) (road l7 l25) (road l7 l65) (road l8 l55) (road l8 l58) (road l9 l28) (road l9 l36) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l7) (vehicle-at l20))
    (:goal (vehicle-at l8))
)