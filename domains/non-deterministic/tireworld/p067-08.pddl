(define (problem tireworld-067-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l54) (road l1 l2) (road l1 l9) (road l10 l27) (road l10 l50) (road l10 l62) (road l11 l30) (road l11 l44) (road l12 l0) (road l12 l62) (road l13 l41) (road l13 l50) (road l14 l34) (road l14 l39) (road l15 l39) (road l15 l60) (road l16 l28) (road l16 l29) (road l17 l59) (road l17 l63) (road l18 l4) (road l18 l51) (road l19 l35) (road l19 l57) (road l2 l1) (road l2 l3) (road l20 l27) (road l20 l4) (road l20 l43) (road l21 l38) (road l21 l60) (road l22 l26) (road l22 l56) (road l23 l24) (road l23 l36) (road l24 l23) (road l25 l48) (road l25 l55) (road l26 l22) (road l26 l45) (road l27 l10) (road l27 l20) (road l28 l16) (road l28 l66) (road l29 l16) (road l29 l8) (road l3 l2) (road l3 l58) (road l30 l11) (road l30 l46) (road l31 l42) (road l31 l66) (road l32 l38) (road l32 l8) (road l33 l40) (road l33 l49) (road l34 l14) (road l34 l5) (road l35 l19) (road l35 l49) (road l36 l23) (road l36 l43) (road l37 l52) (road l37 l63) (road l38 l21) (road l38 l32) (road l39 l14) (road l39 l15) (road l39 l4) (road l4 l18) (road l4 l20) (road l4 l39) (road l40 l33) (road l40 l51) (road l40 l54) (road l41 l13) (road l41 l47) (road l42 l31) (road l42 l55) (road l43 l20) (road l43 l36) (road l43 l44) (road l43 l65) (road l44 l11) (road l44 l43) (road l45 l26) (road l45 l64) (road l46 l30) (road l46 l6) (road l47 l41) (road l47 l53) (road l48 l25) (road l48 l7) (road l49 l33) (road l49 l35) (road l5 l34) (road l5 l54) (road l50 l10) (road l50 l13) (road l50 l52) (road l51 l18) (road l51 l40) (road l52 l37) (road l52 l50) (road l53 l47) (road l53 l9) (road l54 l0) (road l54 l40) (road l54 l5) (road l55 l25) (road l55 l42) (road l55 l6) (road l56 l22) (road l56 l58) (road l57 l19) (road l57 l60) (road l58 l3) (road l58 l56) (road l59 l17) (road l59 l7) (road l6 l46) (road l6 l55) (road l60 l15) (road l60 l21) (road l60 l57) (road l61 l64) (road l61 l65) (road l62 l10) (road l62 l12) (road l63 l17) (road l63 l37) (road l64 l45) (road l64 l61) (road l65 l43) (road l65 l61) (road l66 l28) (road l66 l31) (road l7 l48) (road l7 l59) (road l8 l29) (road l8 l32) (road l9 l1) (road l9 l53) (spare-in l0) (spare-in l1) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l24))
    (:goal (vehicle-at l33))
)