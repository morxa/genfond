(define (problem tireworld-084-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l9 - location)
    (:init (road l0 l13) (road l0 l75) (road l1 l69) (road l1 l78) (road l10 l81) (road l10 l9) (road l11 l56) (road l11 l65) (road l12 l3) (road l12 l66) (road l13 l0) (road l13 l26) (road l14 l46) (road l14 l71) (road l15 l19) (road l15 l4) (road l16 l35) (road l16 l39) (road l17 l23) (road l17 l3) (road l18 l40) (road l18 l55) (road l19 l15) (road l19 l49) (road l2 l21) (road l2 l24) (road l20 l21) (road l20 l45) (road l21 l2) (road l21 l20) (road l22 l34) (road l22 l36) (road l22 l38) (road l22 l54) (road l23 l17) (road l23 l70) (road l24 l2) (road l24 l44) (road l25 l61) (road l25 l8) (road l26 l13) (road l26 l31) (road l27 l53) (road l28 l4) (road l28 l51) (road l29 l57) (road l29 l9) (road l3 l12) (road l3 l17) (road l30 l37) (road l30 l78) (road l31 l26) (road l31 l44) (road l32 l39) (road l32 l58) (road l33 l53) (road l33 l75) (road l34 l22) (road l34 l81) (road l35 l16) (road l35 l72) (road l36 l22) (road l36 l47) (road l37 l30) (road l37 l73) (road l38 l22) (road l38 l68) (road l39 l16) (road l39 l32) (road l4 l15) (road l4 l28) (road l40 l18) (road l40 l64) (road l41 l63) (road l41 l77) (road l42 l5) (road l42 l83) (road l43 l67) (road l43 l77) (road l44 l24) (road l44 l31) (road l45 l20) (road l45 l76) (road l46 l14) (road l46 l52) (road l47 l36) (road l47 l60) (road l48 l62) (road l48 l63) (road l49 l19) (road l49 l59) (road l5 l42) (road l5 l51) (road l50 l62) (road l50 l79) (road l51 l28) (road l51 l5) (road l52 l46) (road l52 l6) (road l53 l27) (road l53 l33) (road l53 l70) (road l54 l22) (road l54 l8) (road l55 l18) (road l55 l73) (road l56 l11) (road l56 l74) (road l57 l29) (road l57 l74) (road l58 l32) (road l58 l7) (road l59 l49) (road l59 l69) (road l6 l52) (road l6 l82) (road l60 l47) (road l60 l70) (road l61 l25) (road l61 l66) (road l62 l48) (road l62 l50) (road l63 l41) (road l63 l48) (road l64 l40) (road l64 l82) (road l65 l11) (road l65 l7) (road l66 l12) (road l66 l61) (road l67 l43) (road l67 l76) (road l68 l38) (road l68 l79) (road l69 l1) (road l69 l59) (road l7 l58) (road l7 l65) (road l70 l23) (road l70 l53) (road l70 l60) (road l71 l14) (road l71 l80) (road l72 l35) (road l72 l83) (road l73 l37) (road l73 l55) (road l74 l56) (road l74 l57) (road l75 l0) (road l75 l33) (road l76 l45) (road l76 l67) (road l77 l41) (road l77 l43) (road l78 l1) (road l78 l30) (road l79 l50) (road l79 l68) (road l8 l25) (road l8 l54) (road l80 l71) (road l81 l10) (road l81 l34) (road l82 l6) (road l82 l64) (road l83 l42) (road l83 l72) (road l9 l10) (road l9 l29) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l9) (vehicle-at l27))
    (:goal (vehicle-at l80))
)