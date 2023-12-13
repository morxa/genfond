(define (problem tireworld-070-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l60) (road l1 l26) (road l1 l35) (road l10 l43) (road l10 l67) (road l11 l18) (road l11 l55) (road l12 l3) (road l12 l56) (road l13 l16) (road l13 l26) (road l14 l23) (road l14 l24) (road l15 l6) (road l15 l65) (road l15 l69) (road l16 l13) (road l16 l58) (road l17 l3) (road l17 l9) (road l18 l11) (road l18 l34) (road l18 l40) (road l19 l0) (road l19 l20) (road l2 l22) (road l2 l25) (road l20 l19) (road l20 l32) (road l21 l54) (road l21 l57) (road l22 l2) (road l22 l41) (road l22 l9) (road l23 l14) (road l23 l32) (road l24 l14) (road l24 l64) (road l25 l2) (road l25 l53) (road l26 l1) (road l26 l13) (road l27 l47) (road l27 l50) (road l28 l63) (road l28 l68) (road l29 l50) (road l29 l69) (road l3 l12) (road l3 l17) (road l30 l58) (road l31 l45) (road l31 l5) (road l32 l20) (road l32 l23) (road l33 l57) (road l34 l18) (road l34 l8) (road l35 l1) (road l35 l37) (road l36 l38) (road l36 l59) (road l37 l35) (road l37 l51) (road l38 l36) (road l38 l46) (road l39 l66) (road l39 l68) (road l4 l41) (road l4 l52) (road l4 l64) (road l40 l18) (road l40 l48) (road l41 l22) (road l41 l4) (road l42 l44) (road l42 l54) (road l43 l10) (road l43 l7) (road l44 l42) (road l44 l56) (road l45 l31) (road l45 l6) (road l46 l38) (road l46 l48) (road l47 l27) (road l47 l59) (road l48 l40) (road l48 l46) (road l49 l53) (road l49 l61) (road l5 l31) (road l5 l67) (road l50 l27) (road l50 l29) (road l51 l37) (road l51 l60) (road l52 l4) (road l52 l63) (road l53 l25) (road l53 l49) (road l54 l21) (road l54 l42) (road l55 l11) (road l55 l66) (road l56 l12) (road l56 l44) (road l57 l21) (road l57 l33) (road l58 l16) (road l58 l30) (road l59 l36) (road l59 l47) (road l6 l15) (road l6 l45) (road l60 l0) (road l60 l51) (road l61 l49) (road l61 l62) (road l62 l61) (road l62 l7) (road l63 l28) (road l63 l52) (road l64 l24) (road l64 l4) (road l65 l15) (road l65 l8) (road l66 l39) (road l66 l55) (road l67 l10) (road l67 l5) (road l68 l28) (road l68 l39) (road l69 l15) (road l69 l29) (road l7 l43) (road l7 l62) (road l8 l34) (road l8 l65) (road l9 l17) (road l9 l22) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l33))
    (:goal (vehicle-at l30))
)