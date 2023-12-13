(define (problem tireworld-085-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l9 - location)
    (:init (road l0 l3) (road l0 l80) (road l1 l3) (road l1 l6) (road l10 l11) (road l10 l57) (road l11 l10) (road l11 l9) (road l12 l36) (road l12 l52) (road l13 l69) (road l13 l83) (road l14 l29) (road l14 l66) (road l15 l44) (road l15 l77) (road l16 l42) (road l16 l47) (road l17 l30) (road l17 l63) (road l18 l4) (road l18 l71) (road l19 l60) (road l19 l77) (road l2 l4) (road l2 l67) (road l20 l54) (road l20 l64) (road l21 l65) (road l22 l43) (road l22 l59) (road l23 l41) (road l23 l46) (road l23 l65) (road l24 l7) (road l24 l8) (road l25 l36) (road l25 l39) (road l26 l49) (road l27 l58) (road l27 l78) (road l28 l50) (road l28 l80) (road l29 l14) (road l29 l45) (road l3 l0) (road l3 l1) (road l30 l17) (road l30 l69) (road l31 l32) (road l31 l81) (road l32 l31) (road l32 l48) (road l33 l35) (road l33 l82) (road l33 l83) (road l34 l57) (road l34 l72) (road l35 l33) (road l35 l75) (road l35 l81) (road l36 l12) (road l36 l25) (road l36 l51) (road l36 l68) (road l37 l46) (road l37 l56) (road l38 l39) (road l38 l61) (road l39 l25) (road l39 l38) (road l4 l18) (road l4 l2) (road l40 l70) (road l40 l81) (road l41 l23) (road l41 l58) (road l42 l16) (road l42 l60) (road l43 l22) (road l43 l63) (road l44 l15) (road l44 l62) (road l45 l29) (road l45 l84) (road l46 l23) (road l46 l37) (road l47 l16) (road l47 l58) (road l48 l32) (road l48 l59) (road l49 l26) (road l49 l73) (road l5 l50) (road l5 l55) (road l50 l28) (road l50 l5) (road l51 l36) (road l51 l75) (road l52 l12) (road l52 l74) (road l53 l70) (road l53 l9) (road l54 l20) (road l54 l60) (road l55 l5) (road l55 l79) (road l56 l37) (road l56 l62) (road l57 l10) (road l57 l34) (road l58 l27) (road l58 l41) (road l58 l47) (road l59 l22) (road l59 l48) (road l6 l1) (road l6 l74) (road l60 l19) (road l60 l42) (road l60 l54) (road l60 l76) (road l60 l79) (road l61 l38) (road l61 l75) (road l62 l44) (road l62 l56) (road l63 l17) (road l63 l43) (road l64 l20) (road l64 l82) (road l65 l21) (road l65 l23) (road l66 l14) (road l66 l68) (road l67 l2) (road l67 l8) (road l68 l36) (road l68 l66) (road l69 l13) (road l69 l30) (road l7 l24) (road l7 l84) (road l70 l40) (road l70 l53) (road l71 l18) (road l71 l78) (road l72 l34) (road l72 l73) (road l73 l49) (road l73 l72) (road l74 l52) (road l74 l6) (road l75 l35) (road l75 l51) (road l75 l61) (road l76 l60) (road l76 l81) (road l77 l15) (road l77 l19) (road l78 l27) (road l78 l71) (road l79 l55) (road l79 l60) (road l8 l24) (road l8 l67) (road l80 l0) (road l80 l28) (road l81 l31) (road l81 l35) (road l81 l40) (road l81 l76) (road l82 l33) (road l82 l64) (road l83 l13) (road l83 l33) (road l84 l45) (road l84 l7) (road l9 l11) (road l9 l53) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l9) (vehicle-at l21))
    (:goal (vehicle-at l26))
)