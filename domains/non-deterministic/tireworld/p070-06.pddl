(define (problem tireworld-070-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l59) (road l1 l29) (road l1 l38) (road l10 l16) (road l10 l39) (road l10 l59) (road l11 l21) (road l11 l3) (road l12 l19) (road l12 l45) (road l13 l42) (road l13 l69) (road l14 l24) (road l14 l48) (road l15 l38) (road l15 l63) (road l16 l10) (road l16 l49) (road l17 l55) (road l17 l57) (road l18 l31) (road l18 l49) (road l19 l12) (road l19 l3) (road l2 l37) (road l2 l67) (road l20 l22) (road l20 l35) (road l21 l0) (road l21 l11) (road l22 l20) (road l22 l53) (road l23 l42) (road l23 l50) (road l24 l14) (road l24 l32) (road l25 l69) (road l25 l7) (road l26 l61) (road l26 l68) (road l27 l51) (road l27 l53) (road l28 l29) (road l28 l44) (road l29 l1) (road l29 l28) (road l3 l11) (road l3 l19) (road l30 l40) (road l30 l60) (road l30 l63) (road l31 l18) (road l31 l56) (road l32 l24) (road l33 l43) (road l33 l47) (road l34 l54) (road l34 l66) (road l35 l20) (road l35 l52) (road l36 l46) (road l36 l52) (road l37 l2) (road l37 l5) (road l38 l1) (road l38 l15) (road l39 l10) (road l39 l64) (road l4 l5) (road l4 l50) (road l40 l30) (road l40 l62) (road l41 l44) (road l41 l8) (road l42 l13) (road l42 l23) (road l43 l33) (road l43 l69) (road l44 l28) (road l44 l41) (road l45 l12) (road l46 l36) (road l46 l68) (road l47 l33) (road l47 l8) (road l48 l14) (road l48 l51) (road l48 l57) (road l49 l16) (road l49 l18) (road l5 l37) (road l5 l4) (road l50 l23) (road l50 l4) (road l50 l61) (road l51 l27) (road l51 l48) (road l52 l35) (road l52 l36) (road l53 l22) (road l53 l27) (road l54 l34) (road l54 l64) (road l55 l17) (road l55 l6) (road l56 l31) (road l56 l58) (road l57 l17) (road l57 l48) (road l58 l56) (road l58 l63) (road l59 l0) (road l59 l10) (road l6 l55) (road l6 l9) (road l60 l30) (road l60 l65) (road l61 l26) (road l61 l50) (road l62 l40) (road l62 l65) (road l63 l15) (road l63 l30) (road l63 l58) (road l64 l39) (road l64 l54) (road l65 l60) (road l65 l62) (road l65 l9) (road l66 l34) (road l66 l67) (road l67 l2) (road l67 l66) (road l68 l26) (road l68 l46) (road l68 l7) (road l69 l13) (road l69 l25) (road l69 l43) (road l7 l25) (road l7 l68) (road l8 l41) (road l8 l47) (road l9 l6) (road l9 l65) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l46) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l59) (spare-in l61) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (vehicle-at l32))
    (:goal (vehicle-at l45))
)