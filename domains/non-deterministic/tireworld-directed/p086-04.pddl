(define (problem tireworld-086-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l9 - location)
    (:init (road l0 l84) (road l1 l22) (road l1 l34) (road l10 l14) (road l10 l41) (road l11 l39) (road l11 l51) (road l12 l27) (road l13 l50) (road l13 l81) (road l14 l10) (road l14 l48) (road l15 l32) (road l15 l76) (road l16 l63) (road l17 l54) (road l17 l85) (road l19 l52) (road l19 l60) (road l2 l21) (road l2 l49) (road l20 l18) (road l20 l26) (road l21 l2) (road l21 l59) (road l22 l1) (road l22 l24) (road l23 l35) (road l23 l70) (road l24 l22) (road l24 l58) (road l25 l69) (road l25 l84) (road l26 l20) (road l26 l59) (road l27 l60) (road l28 l46) (road l28 l61) (road l29 l3) (road l29 l54) (road l3 l29) (road l3 l83) (road l30 l57) (road l30 l62) (road l31 l32) (road l31 l55) (road l32 l15) (road l32 l31) (road l32 l36) (road l33 l72) (road l34 l1) (road l35 l23) (road l35 l49) (road l35 l61) (road l36 l32) (road l37 l41) (road l38 l56) (road l38 l77) (road l39 l11) (road l39 l8) (road l4 l66) (road l40 l68) (road l41 l10) (road l41 l37) (road l42 l43) (road l43 l74) (road l44 l5) (road l45 l53) (road l45 l72) (road l46 l28) (road l46 l82) (road l47 l77) (road l48 l14) (road l48 l71) (road l49 l2) (road l49 l35) (road l5 l44) (road l5 l51) (road l50 l6) (road l51 l11) (road l51 l5) (road l52 l0) (road l53 l78) (road l54 l29) (road l55 l31) (road l55 l67) (road l56 l9) (road l57 l30) (road l57 l75) (road l58 l16) (road l58 l24) (road l59 l21) (road l59 l26) (road l6 l27) (road l6 l50) (road l6 l64) (road l60 l19) (road l60 l27) (road l61 l28) (road l61 l35) (road l62 l30) (road l63 l12) (road l63 l16) (road l63 l42) (road l64 l6) (road l65 l73) (road l65 l76) (road l66 l33) (road l66 l4) (road l67 l76) (road l68 l4) (road l68 l40) (road l69 l25) (road l69 l34) (road l7 l81) (road l7 l9) (road l70 l23) (road l70 l74) (road l71 l40) (road l71 l48) (road l72 l45) (road l73 l85) (road l74 l70) (road l75 l64) (road l76 l15) (road l76 l65) (road l76 l67) (road l77 l38) (road l77 l47) (road l78 l62) (road l79 l44) (road l79 l82) (road l8 l80) (road l80 l36) (road l80 l8) (road l81 l13) (road l82 l79) (road l83 l3) (road l83 l37) (road l84 l0) (road l84 l25) (road l85 l17) (road l85 l73) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l52) (spare-in l54) (spare-in l56) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l9) (vehicle-at l47))
    (:goal (vehicle-at l18))
)