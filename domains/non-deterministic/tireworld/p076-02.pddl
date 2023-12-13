(define (problem tireworld-076-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l66) (road l0 l8) (road l1 l12) (road l1 l18) (road l10 l8) (road l10 l9) (road l11 l16) (road l11 l48) (road l12 l1) (road l12 l61) (road l13 l42) (road l13 l43) (road l14 l17) (road l14 l66) (road l15 l22) (road l15 l31) (road l16 l11) (road l16 l72) (road l17 l14) (road l17 l6) (road l18 l1) (road l18 l57) (road l19 l37) (road l19 l53) (road l2 l29) (road l2 l41) (road l20 l33) (road l20 l63) (road l21 l42) (road l21 l49) (road l22 l15) (road l22 l39) (road l23 l44) (road l23 l55) (road l24 l46) (road l24 l73) (road l25 l51) (road l25 l64) (road l26 l30) (road l26 l50) (road l27 l59) (road l27 l71) (road l28 l52) (road l28 l68) (road l29 l2) (road l29 l58) (road l3 l34) (road l3 l5) (road l3 l75) (road l30 l26) (road l30 l49) (road l31 l15) (road l31 l60) (road l32 l58) (road l32 l59) (road l33 l20) (road l33 l34) (road l34 l3) (road l34 l33) (road l35 l41) (road l35 l51) (road l36 l53) (road l36 l61) (road l37 l19) (road l37 l67) (road l38 l65) (road l38 l72) (road l39 l22) (road l39 l50) (road l39 l74) (road l4 l69) (road l4 l71) (road l40 l45) (road l40 l74) (road l41 l2) (road l41 l35) (road l42 l13) (road l42 l21) (road l43 l13) (road l43 l44) (road l44 l23) (road l44 l43) (road l45 l40) (road l45 l7) (road l45 l75) (road l46 l24) (road l46 l62) (road l47 l60) (road l47 l62) (road l48 l11) (road l48 l7) (road l49 l21) (road l49 l30) (road l5 l3) (road l5 l75) (road l50 l26) (road l50 l39) (road l50 l63) (road l51 l25) (road l51 l35) (road l52 l28) (road l52 l65) (road l53 l19) (road l53 l36) (road l54 l68) (road l54 l70) (road l55 l23) (road l55 l64) (road l56 l57) (road l56 l70) (road l57 l18) (road l57 l56) (road l58 l29) (road l58 l32) (road l59 l27) (road l59 l32) (road l6 l17) (road l6 l69) (road l60 l31) (road l60 l47) (road l61 l12) (road l61 l36) (road l62 l46) (road l62 l47) (road l63 l20) (road l63 l50) (road l64 l25) (road l64 l55) (road l65 l38) (road l65 l52) (road l66 l0) (road l66 l14) (road l67 l37) (road l67 l9) (road l68 l28) (road l68 l54) (road l69 l4) (road l69 l6) (road l7 l45) (road l7 l48) (road l70 l54) (road l70 l56) (road l71 l27) (road l71 l4) (road l72 l16) (road l72 l38) (road l73 l24) (road l73 l9) (road l74 l39) (road l74 l40) (road l75 l3) (road l75 l45) (road l75 l5) (road l8 l0) (road l8 l10) (road l9 l10) (road l9 l67) (road l9 l73) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l11))
)